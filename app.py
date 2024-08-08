import os
import cv2
import pymysql.cursors
from flask import Flask, Response, request, render_template, redirect, url_for, jsonify
from inference import get_model
import supervision as sv
from werkzeug.utils import secure_filename
import random
import datetime

#Roboflow API key
API_KEY = "DeRtV2c0IBAZ71gHnBey"

# Load a pre-trained YOLOv8 model
model = get_model(model_id="with-vehicle/1", api_key=API_KEY)

# Create supervision annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

input_video_path = None
latest_predictions = []
sector = None  # Initialize sector variable

# MySQL database connection settings
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='0108',
    database='wildlife',
    cursorclass=pymysql.cursors.DictCursor
)

def get_random_sector():
    # Generate a random number between 1 and 100 for sectors
    sector_number = random.randint(1, 100)
    return f"SEC {sector_number}"

def save_predictions_to_db(predictions):
    try:
        with connection.cursor() as cursor:
            for prediction in predictions:
                # Check if the same label has been detected within the last 3 minutes
                sql_check = """
                    SELECT * FROM detections 
                    WHERE detected_label = %s 
                    AND sector = %s 
                    AND timestamp > NOW() - INTERVAL 3 MINUTE
                """
                cursor.execute(sql_check, (prediction['label'], prediction['sector']))
                recent_detection = cursor.fetchone()

                if not recent_detection:
                    
                    sql_insert = "INSERT INTO detections (detected_label, confidence, sector, timestamp) VALUES (%s, %s, %s, %s)"
                    cursor.execute(sql_insert, (prediction['label'], prediction['confidence'], prediction['sector'], datetime.datetime.now()))
            connection.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")

# def generate_frames(input_video_path):
#     global latest_predictions
#     global sector

#     cap = cv2.VideoCapture(input_video_path)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Resize the frame to 720p
#         frame = cv2.resize(frame, (1280, 720))
        
#         results = model.infer(frame)[0]
#         # Check if predictions are available
#         if results.predictions:
#             latest_predictions = [{"label": prediction.class_name, "confidence": prediction.confidence, "sector": sector} for prediction in results.predictions]
#             save_predictions_to_db(latest_predictions)  # Save to database
#         else:
#             latest_predictions = []
#         detections = sv.Detections.from_inference(results)
#         annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
#         annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
#         ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
#         if not ret:
#             continue
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     cap.release()



def generate_frames(input_video_path):
    global latest_predictions
    global sector

    cap = cv2.VideoCapture(input_video_path)
    frame_skip = 2  # Process every 2nd frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames based on frame_skip value

        # Resize the frame to a higher resolution (e.g., 1280x720)
        frame = cv2.resize(frame, (1280, 720))

        results = model.infer(frame)[0]
        if results.predictions:
            latest_predictions = [{"label": prediction.class_name, "confidence": prediction.confidence, "sector": sector} for prediction in results.predictions]
            save_predictions_to_db(latest_predictions)
        else:
            latest_predictions = []
        detections = sv.Detections.from_inference(results)
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Encode the frame and yield it to the video feed
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    global input_video_path
    global sector
    
    if 'videoFile' not in request.files:
        return 'No file part', 400
    file = request.files['videoFile']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        input_video_path = file_path
        
        sector = get_random_sector()
        
        return 'File uploaded successfully', 200
    return 'File upload failed', 400

@app.route('/play')
def play():
    return render_template('play.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(input_video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_predictions')
def get_latest_predictions():
    return jsonify(latest_predictions)

@app.route('/fire')
def fire():
    return render_template('fire.html')

@app.route('/poaching')
def poaching():
    return render_template('poaching.html')

@app.route('/behaviour')
def behaviour():
    return render_template('behaviour.html')

@app.route('/treefalls')
def treefalls():
    return render_template('treefalls.html')


@app.route('/blacklisted', methods=['GET', 'POST'])
def blacklisted_vehicles():
    if request.method == 'POST':
        vehicle_number = request.form.get('vehicle_number')
        sector = request.form.get('sector')
        
        try:
            with connection.cursor() as cursor:
                sql = "INSERT INTO blacklisted_vehicles (vehicle_number, sector) VALUES (%s, %s)"
                cursor.execute(sql, (vehicle_number, sector))
                connection.commit()
        except Exception as e:
            print(f"Error inserting into database: {e}")
    
    vehicles = []
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM blacklisted_vehicles"
            cursor.execute(sql)
            vehicles = cursor.fetchall()
    except Exception as e:
        print(f"Error fetching from database: {e}")
    
    return render_template('blacklisted.html', vehicles=vehicles)

@app.route('/delete_vehicle/<int:vehicle_id>', methods=['POST'])
def delete_vehicle(vehicle_id):
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM blacklisted_vehicles WHERE id = %s"
            cursor.execute(sql, (vehicle_id,))
            connection.commit()
    except Exception as e:
        print(f"Error deleting from database: {e}")
    
    return redirect(url_for('blacklisted_vehicles'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


































































































#print for every secs

# import os
# import cv2
# from flask import Flask, Response, request, render_template, jsonify
# from inference import get_model
# import supervision as sv
# from werkzeug.utils import secure_filename

# # Your Roboflow API key
# API_KEY = "rQ5TGMhiCaT0WP8Y0p6u"

# # Load a pre-trained YOLOv8 model
# model = get_model(model_id="tiger-deer-poc/1", api_key=API_KEY)

# # Create supervision annotators
# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# input_video_path = None
# latest_predictions = []
# video_uploaded = False  # Flag to check if video is uploaded

# def generate_frames(input_video_path):
#     global latest_predictions
#     global video_uploaded
#     video_uploaded = True
#     # Open the video file
#     cap = cv2.VideoCapture(input_video_path)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Resize the frame to 720p
#         frame = cv2.resize(frame, (1280, 720))
#         # Run inference on the current frame
#         results = model.infer(frame)[0]
#         # Check if predictions are available
#         if results.predictions:
#             latest_predictions = [{"label": prediction.class_name, "confidence": prediction.confidence} for prediction in results.predictions]
#         else:
#             latest_predictions = []
#         # Load the results into the supervision Detections API
#         detections = sv.Detections.from_inference(results)
#         # Annotate the frame with bounding boxes and labels
#         annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
#         annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
#         # Encode frame as JPEG with lower quality for faster streaming
#         ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
#         if not ret:
#             continue
#         # Convert to bytes
#         frame_bytes = buffer.tobytes()
#         # Yield frame in HTTP response format
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     cap.release()
#     video_uploaded = False  # Reset the flag when video processing is done

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def handle_upload():
#     global input_video_path
#     global video_uploaded
#     if 'videoFile' not in request.files:
#         return 'No file part', 400
#     file = request.files['videoFile']
#     if file.filename == '':
#         return 'No selected file', 400
#     if file:
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#         input_video_path = file_path
#         video_uploaded = True  # Set the flag to True when a video is uploaded
#         return 'File uploaded successfully', 200
#     return 'File upload failed', 400

# @app.route('/play')
# def play():
#     return render_template('play.html')

# @app.route('/video_feed')
# def video_feed():
#     if video_uploaded:  # Ensure video is uploaded before streaming
#         return Response(generate_frames(input_video_path),
#                         mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return 'No video uploaded', 404

# @app.route('/latest_predictions')
# def get_latest_predictions():
#     if video_uploaded:
#         return jsonify(latest_predictions)
#     else:
#         return jsonify({"message": "No video is currently being processed."}), 404

# @app.route('/fire')
# def fire():
#     return render_template('fire.html')

# @app.route('/poaching')
# def poaching():
#     return render_template('poaching.html')

# @app.route('/behaviour')
# def behaviour():
#     return render_template('behaviour.html')

# @app.route('/treefalls')
# def treefalls():
#     return render_template('treefalls.html')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
