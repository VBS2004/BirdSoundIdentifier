from flask import Blueprint, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
from models import User  # Ensure models.py has the User model defined
from models import db  # Ensure models.py has the db instance
import re

bycrypt = Bcrypt()
auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def signup():
    data=request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"msg": "Invalid email format"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"msg": "Email already registered"}), 400
    
    hashed_pw=bycrypt.generate_password_hash(password).decode('utf-8')

    user=User(email=email,password=hashed_pw)
    db.session.add(user)
    db.session.commit()

    return jsonify({"msg": "User created successfully"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data=request.get_json()
    email=data.get('email')
    password=data.get('password')

    user=User.query.filter_by(email=email).first()
    if not user or not bycrypt.check_password_hash(user.password, password):
        return jsonify({"msg": "Invalid credentials"}), 401
    
    token=create_access_token(identity=user.id,expires_delta=timedelta(days=1))

    return jsonify({"token":token,"userId":user.id}),200
@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    user_id=get_jwt_identity()
    user=User.query.get(user_id)

    return jsonify({"id":user.id,"email":user.email}),200