from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from predict import *
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        return send_from_directory('downloaded_images', filename)
    except FileNotFoundError:
        return {"error": "Image not found"}, 404

@app.route("/")
def home():
    return {"message": "Hello from backend"}
@app.route("/species", methods=['GET'])
def get_species():
    species = get_species_list()
    return jsonify(species)
@app.route("/upload", methods=['POST'])
def upload():
    file = request.files['file']
    file.save('uploads/' + file.filename)

    # Load the image to predict
    prediction, CV = predict(f"V:\\BirdSoundIdentifier\\uploads\\{file.filename}")

    params = {
        "engine": "google_images",
        "q": prediction,  # Use the actual prediction instead of hardcoded value
        "api_key": "3759820c47efbd8d36b8e91f2d0e41350c4cfa9839ca66370e80c6c382ea60d3"
    }

    response = requests.get("https://serpapi.com/search", params=params)

    # Create folder using filename without extension
    folder_name = os.path.splitext(file.filename)[0]
    os.makedirs(f"downloaded_images/{folder_name}", exist_ok=True)
    
    count = 0
    downloaded_images = []  # This is the key addition!

    for idx, image in enumerate(response.json().get('images_results', [])):
        image_url = image.get('original', '')
        print(f"Processing image {idx}: {image_url}")
        
        if (image_url.lower().endswith(".jpeg") or 
            image_url.lower().endswith(".jpg") or 
            image_url.lower().endswith(".png")) and count < 10:
            try:
                img_data = requests.get(image_url, timeout=10).content
                extension = image_url.split(".")[-1]
                
                # Save the image
                image_filename = f"image_{idx}.{extension}"
                filepath = f"downloaded_images/{folder_name}/{image_filename}"
                
                with open(filepath, "wb") as f:
                    f.write(img_data)
                
                # Add to the list of successfully downloaded images
                downloaded_images.append(f"images/{folder_name}/{image_filename}")
                
                print(f"Downloaded: {image_filename}")
                count += 1
                
            except Exception as e:
                print(f"Failed to download image {idx}: {e}")
                continue

    print(f"Total images downloaded: {len(downloaded_images)}")
    print(f"Downloaded image URLs: {downloaded_images}")

    return jsonify({
        "filename": file.filename,
        "status": "success",
        "prediction": prediction,
        "confidence": CV,
        "image_location": f"downloaded_images/{folder_name}/",
        "image_urls": downloaded_images  # Add this crucial field!
    })

if __name__ == '__main__':
    app.run(debug=True)