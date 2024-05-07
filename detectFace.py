from flask import Flask, request, jsonify
import os
# import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# from keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
app = Flask(__name__)
from tensorflow import keras
import glob
from flask_cors import CORS

CORS(app)
import os

# Specify the folder path
folder_path = "faces"

# Initialize a counter variable

rootdir = "faces"
pattern = os.path.join(rootdir, "*.jpg")
e2tcModel = load_model('original.h5')
def preprocess_image(image_path):
  image = load_img(image_path, target_size=(224, 224))
  image = img_to_array(image)
  print(image.shape)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  return image
@app.route('/predictFace', methods=['GET'])
def process_video():
    print("Hello world")
    count=0
    file_count = 0

# Iterate through all entries in the folder
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)

        if os.path.isfile(full_path):
            file_count += 1
    
    print(file_count)
    for filename in glob.glob(pattern):
        preprocessed_image = preprocess_image(filename)
        prediction = e2tcModel.predict(preprocessed_image)
        print(prediction)
        if prediction <= 0.5:
            count+=1
            # return jsonify({'prediction': "Fake"})
        if count/file_count >= 0.33:
            return jsonify({'prediction': "Fake"})
    print(count)

    return jsonify({'prediction': "Real"})


if __name__ == '__main__':
    app.run(debug=True, port=8000)