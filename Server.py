from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from predict import *

# # Importing deps for image prediction
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import numpy as np
# from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/")
def home():
    return {"message": "Hello from backend"}

@app.route("/upload", methods=['POST'])
def upload():
    file = request.files['file']
    file.save('uploads/' + file.filename)

    # Load the image to predict
    img_path = rf"V:\BirdSoundIdentifier\uploads\{file.filename}"
    
    prediction,CV=predict(f"V:\\BirdSoundIdentifier\\uploads\\{file.filename}")

    return jsonify({
            "filename": file.filename,
            "status": "success",
            "prediction":prediction,
            "confidence":CV
        })

if __name__ == '__main__':
    app.run(debug=True)