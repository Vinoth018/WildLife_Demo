def generate_frames(input_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on the current frame
        results = model.infer(frame)[0]
        
        # Print the structure of results to understand it
        print("Inference results:", results)  # Inspect this line to understand the structure
        
        # Extract labels from results
        # Assuming results is a list of tuples or lists where each tuple/list contains labels
        for detection in results:
            # Assuming detection is a tuple or list where the label is at index 0
            # Adjust index based on actual structure
            label = detection[0]  # Change this based on actual index or structure
            print(f"Detected label: {label}")
        
        # Load the results into the supervision Detections API
        detections = sv.Detections.from_inference(results)
        
        # Annotate the frame with bounding boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        # Convert to bytes
        frame_bytes = buffer.tobytes()

        # Yield frame in HTTP response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()























    # def generate_frames(input_video_path):
#     # Open the video file
#     cap = cv2.VideoCapture(input_video_path)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Run inference on the current frame
#         results = model.infer(frame)[0]
        
#         # Load the results into the supervision Detections API
#         detections = sv.Detections.from_inference(results)
        
#         # Annotate the frame with bounding boxes and labels
#         annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
#         annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
#         # Encode frame as JPEG
#         ret, buffer = cv2.imencode('.jpg', annotated_frame)
#         if not ret:
#             continue

#         # Convert to bytes
#         frame_bytes = buffer.tobytes()

#         # Yield frame in HTTP response format
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

























def generate_frames(input_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to 720p
        frame = cv2.resize(frame, (1280, 720))
        # Run inference on the current frame
        results = model.infer(frame)[0]
        # Load the results into the supervision Detections API
        detections = sv.Detections.from_inference(results)
        # Annotate the frame with bounding boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        # Encode frame as JPEG with lower quality for faster streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        # Convert to bytes
        frame_bytes = buffer.tobytes()
        # Yield frame in HTTP response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')
@app.route('/upload', methods=['POST'])
def handle_upload():
    global input_video_path
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
        return 'File uploaded successfully', 200
    return 'File upload failed', 400