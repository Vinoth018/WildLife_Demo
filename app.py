
import os
import cv2
import pymysql.cursors
from flask import Flask, Response, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import random
import datetime
from ultralytics import YOLO  # Import YOLO from ultralytics
from inference_sdk import InferenceHTTPClient
import json
import easyocr

ocr_reader = easyocr.Reader(['en'])

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

input_video_path = None
latest_predictions = []
sector = None  # Initialize sector variable

# Load the pre-trained YOLOv8 model from the local file
model = YOLO(r'C:\Users\vinothg\Desktop\Wild\TELANGANA MODEL POC.pt')

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="DeRtV2c0IBAZ71gHnBey"
)

ocr_results_file = os.path.join(output_dir, 'ocr_results.json')

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

        # Perform detection using the YOLOv8 model
        results = model(frame)

        latest_predictions = []
        for result in results:
            for box in result.boxes:
                latest_predictions.append({
                    "label": result.names[int(box.cls)],
                    "confidence": box.conf.item(),
                    "sector": sector
                })
            save_predictions_to_db(latest_predictions)

        # Annotate the frame with the detections
        for result in results:
            frame = result.plot()

        # Encode the frame and yield it to the video feed
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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
        
        sector = get_random_sector()  # Assign a new sector number for this video
        
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

@app.route('/vehicle')
def vehicle_index():
    return render_template('vehicle_index.html')

@app.route('/uploads', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))
    
    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(url_for('index'))
    
    video_path = os.path.join(output_dir, video_file.filename)
    video_file.save(video_path)

    # Process the video and save frames
    process_video(video_path)

    return redirect(url_for('results'))

def process_video(video_path):
    global sector
    sector = get_random_sector()  # Generate a new sector number for each new video
    video_capture = cv2.VideoCapture(video_path)
    frame_number = 0
    all_results = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Resize the frame to reduce its size
        frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 pixels
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
        cv2.imwrite(frame_path, frame_rgb)
        
        # Perform inference to detect number plates
        result = CLIENT.infer(frame_path, model_id="car-plate-detection-sctyn/3")
        detections = parse_result(result)
        draw_bounding_boxes(frame_rgb, detections)
        
        # Save the labeled frame
        labeled_frame_path = os.path.join(output_dir, f"labeled_frame_{frame_number}.jpg")
        cv2.imwrite(labeled_frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        # Perform OCR on the detected areas and collect results, passing sector as an argument
        ocr_results = perform_ocr_on_cropped_images(detections, frame_rgb, frame_number, sector)
        all_results.extend(ocr_results)

        frame_number += 1

    video_capture.release()

    # Save OCR results to a JSON file
    with open(ocr_results_file, 'w') as f:
        json.dump(all_results, f)

def parse_result(result):
    detections = []
    for prediction in result['predictions']:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        confidence = prediction['confidence']
        class_name = prediction['class']
        x_min = int(x - width / 2)
        y_min = int(y - height / 2)
        x_max = int(x + width / 2)
        y_max = int(y + width / 2)
        detections.append((x_min, y_min, x_max, y_max, confidence, class_name))
    return detections

def draw_bounding_boxes(image, detections):
    for (x_min, y_min, x_max, y_max, confidence, class_name) in detections:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} ({confidence:.2f})", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def perform_ocr_on_cropped_images(detections, image_rgb, frame_number, sector):
    results = []
    for idx, (x_min, y_min, x_max, y_max, confidence, class_name) in enumerate(detections):
        cropped_image = image_rgb[y_min:y_max, x_min:x_max]
        ocr_results = ocr_reader.readtext(cropped_image)
        for bbox, text, score in ocr_results:
            # Append the sector number to the detected text
            detected_text_with_sector = f"{text} and {sector}"
            results.append({
                'frame': frame_number,
                'text': detected_text_with_sector,
                'confidence': score
            })
            # print(f"Detected text '{detected_text_with_sector}' with confidence {score:.2f}")
    return results


@app.route('/results')
def results():
    # List all labeled frames
    labeled_frames = sorted([f for f in os.listdir(output_dir) if f.startswith('labeled_frame_')])
    
    # Load OCR results
    if os.path.exists(ocr_results_file):
        with open(ocr_results_file, 'r') as f:
            ocr_results = json.load(f)
    else:
        ocr_results = []
    
    return render_template('results.html', frames=labeled_frames, ocr_results=ocr_results)

@app.route('/frames/<path:filename>')
def frames(filename):
    return send_from_directory(output_dir, filename)

@app.route('/ocr_results')
def ocr_results():
    if os.path.exists(ocr_results_file):
        with open(ocr_results_file, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)