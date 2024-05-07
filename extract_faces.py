from flask import Flask, request, jsonify, send_file
import cv2
import os
import cv2
import numpy as  np
import matplotlib.pyplot as plt
from tensorflow import keras
from flask_cors import CORS
from retinaface import RetinaFace
import shutil
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
import json
import cv2  # OpenCV library for image processing
from PIL import Image
#Loading model with json

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
# Replace 'path/to/model.h5' with the actual path to your saved model file
@app.route('/extractVideo', methods=['POST'])
def extractVideo():
    # Check if the POST request has the file part
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    # Check if the file is an allowed extension
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format'})
    
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    filename=file.filename
    # Save uploaded file
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)
    # Use VideoCapture to read frames from the video
    video_capture = cv2.VideoCapture(video_path)
    # Initialize frame count
    frame_count = 0
    # Read a single frame from the video
    faces_folder = 'faces'
    if os.path.exists(faces_folder):
      shutil.rmtree(faces_folder)
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
    else:
        while True:
        # Read a single frame from the video
            ret, frame = video_capture.read()
            
            # Break the loop if no more frames to read
            if not ret:
                break
            
            # Increment frame count
            frame_count += 1
            # Detect faces in the frame
            faces = RetinaFace.detect_faces(frame)
            # Iterate through each detected face
            for face_key, face_data in faces.items():
                
                #Extract the facial area coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = face_data['facial_area']
                
                # Crop the image based on the coordinates
                cropped_face = frame[y1:y2, x1:x2]

                output_path = os.path.join(faces_folder, f"{filename}_frame_{frame_count}_face_{face_key}.jpg")
                cv2.imwrite(output_path, cropped_face)
                
    return jsonify({"Message": "Extracted and saved"})

if __name__ == '__main__':
    app.run(debug=True)

   
  