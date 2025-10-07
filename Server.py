from flask_jwt_extended import JWTManager,jwt_required,get_jwt_identity
from auth import auth_bp
from models import db,History
from flask_bcrypt import Bcrypt
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from predict import predict, get_species_list  # Make sure predict.py is correct
import redis
import json

# Your Pexels API Key - replace with your actual key
PEXELS_API_KEY = "TfSSEu0w6McJiyIfu1o5AYp6rtO7OBbl1AyXT5SFDnPDQhynh5bPfXLN"

# In-memory cache: store responses for 24 hours (86400 seconds)
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True  # Automatically decode byte responses to strings
)
CACHE_TTL = 86400


def search_pexels_images(query, per_page=50):
    """
    Search for images using Pexels API
    """
    headers = {
        'Authorization': PEXELS_API_KEY
    }
    
    params = {
        'query': query,
        'per_page': min(per_page, 80),  # Pexels max is 80 per page
        'page': 1
    }
    
    try:
        response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching images from Pexels: {e}")
        return None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///birdsounds.db'
app.config['JWT_SECRET_KEY']='super-secret'

db.init_app(app)
jwt=JWTManager(app)

app.register_blueprint(auth_bp, url_prefix='/api/auth')

with app.app_context():
    db.create_all()
@app.route("/")
def home():
    return {"message": "Hello from backend"}

@app.route("/species/images", methods=['POST'])
def get_species_images():
    data = request.get_json()
    if not data or 'species_name' not in data:
        return jsonify({"error": "species_name is required"}), 400

    species_name = data['species_name']
    if not species_name:
        return jsonify({"error": "species_name cannot be empty"}), 400

    # Check cache first
    cache_key = f"images:{species_name}"
    try:
        cached_response = redis_client.get(cache_key)
        if cached_response:
            print(f"Cache hit for {species_name}")
            return jsonify(json.loads(cached_response))
    except redis.RedisError as e:
        print(f"Redis error: {e}")

    # Search for images using Pexels API
    search_results = search_pexels_images(f"{species_name} bird", per_page=10)  # Limit to 10 to avoid rate limits
    
    if not search_results or 'photos' not in search_results:
        return jsonify({"error": f"No images found for {species_name}"}), 500

    image_urls = []
    for photo in search_results['photos']:
        try:
            photo_urls = {
                'id': photo['id'],
                'photographer': photo['photographer'],
                'alt': photo['alt'],
                'original': photo['src']['original'],
                'large2x': photo['src']['large2x'],
                'large': photo['src']['large'],
                'medium': photo['src']['medium'],
                'small': photo['src']['small'],
                'portrait': photo['src']['portrait'],
                'landscape': photo['src']['landscape'],
                'tiny': photo['src']['tiny']
            }
            image_urls.append(photo_urls)
        except Exception as e:
            print(f"Failed to process image {photo.get('id', 'unknown')}: {e}")
            continue

    response = {
        "status": "success",
        "species_name": species_name,
        "total_images": len(image_urls),
        "image_urls": image_urls
    }

    # Store in Redis with TTL
    try:
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(response))
        print(f"Cached response for {species_name}")
    except redis.RedisError as e:
        print(f"Redis error on set: {e}")
        
    return jsonify(response)

@app.route("/species", methods=['GET'])
def get_species():
    species = get_species_list()
    species_list = []
    count = 1

    for bird_name in species:
        species_list.append({
            "id": count,
            "name": bird_name
        })
        count += 1

    return jsonify({"species": species_list})

@app.route("/upload", methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    # Save uploaded file
    uploads_dir = os.path.abspath("uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Predict bird species
    prediction, CV = predict(file_path)

    # Search for images using Pexels API
    search_results = search_pexels_images(f"{prediction} bird", per_page=80)  # Pexels max is 80 per page
    
    if not search_results or 'photos' not in search_results:
        return jsonify({"error": "No images found for the predicted species"}), 500

    # Collect image URLs without downloading
    image_urls = []
    
    for photo in search_results['photos']:
        if len(image_urls) >= 100:  # Limit to 100 images
            break
            
        try:
            photo_urls = {
                'id': photo['id'],
                'photographer': photo['photographer'],
                'alt': photo['alt'],
                'original': photo['src']['original'],
                'large2x': photo['src']['large2x'],
                'large': photo['src']['large'],
                'medium': photo['src']['medium'],
                'small': photo['src']['small'],
                'portrait': photo['src']['portrait'],
                'landscape': photo['src']['landscape'],
                'tiny': photo['src']['tiny']
            }
            
            image_urls.append(photo_urls)
                
        except Exception as e:
            print(f"Failed to process image {photo.get('id', 'unknown')}: {e}")
            continue

    # If we need more images and didn't reach 100, try a second page
    if len(image_urls) < 100 and len(search_results['photos']) == 80:
        try:
            headers = {'Authorization': PEXELS_API_KEY}
            params = {
                'query': f"{prediction} bird",
                'per_page': min(100 - len(image_urls), 80),
                'page': 2
            }
            
            response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)
            if response.status_code == 200:
                page2_results = response.json()
                
                for photo in page2_results.get('photos', []):
                    if len(image_urls) >= 100:
                        break
                        
                    try:
                        photo_urls = {
                            'id': photo['id'],
                            'photographer': photo['photographer'],
                            'alt': photo['alt'],
                            'original': photo['src']['original'],
                            'large2x': photo['src']['large2x'],
                            'large': photo['src']['large'],
                            'medium': photo['src']['medium'],
                            'small': photo['src']['small'],
                            'portrait': photo['src']['portrait'],
                            'landscape': photo['src']['landscape'],
                            'tiny': photo['src']['tiny']
                        }
                        
                        image_urls.append(photo_urls)
                            
                    except Exception as e:
                        print(f"Failed to process image {photo.get('id', 'unknown')} from page 2: {e}")
                        continue
                        
        except Exception as e:
            print(f"Failed to fetch second page: {e}")

    return jsonify({
        "filename": file.filename,
        "status": "success",
        "prediction": prediction,
        "confidence": CV,
        "total_images": len(image_urls),
        "image_urls": image_urls
    })

if __name__ == '__main__':
    app.run(debug=True)