"""
Simple Authentication Service for SPOKHAND SIGNCUT
Handles basic authentication for local testing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import bcrypt
import uuid
from datetime import datetime, timedelta
import os

# Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Simple in-memory user storage for testing
users = {}

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_jwt(user_id: str, email: str, roles: list) -> str:
    """Generate a JWT token"""
    payload = {
        'user_id': user_id,
        'email': email,
        'roles': roles,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt(token: str) -> dict:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        if email in users:
            return jsonify({'error': 'User already exists'}), 400
        
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(password)
        
        users[email] = {
            'user_id': user_id,
            'email': email,
            'password': hashed_password,
            'full_name': full_name,
            'roles': ['user'],
            'created_at': datetime.utcnow().isoformat()
        }
        
        token = generate_jwt(user_id, email, ['user'])
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': user_id,
            'token': token
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login a user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        if email not in users:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = users[email]
        if not verify_password(password, user['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        token = generate_jwt(user['user_id'], email, user['roles'])
        
        return jsonify({
            'message': 'Login successful',
            'user_id': user['user_id'],
            'email': email,
            'roles': user['roles'],
            'token': token
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/verify', methods=['POST'])
def verify_token():
    """Verify a JWT token"""
    try:
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'error': 'Token is required'}), 400
        
        payload = verify_jwt(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        return jsonify({
            'valid': True,
            'user_id': payload['user_id'],
            'email': payload['email'],
            'roles': payload['roles']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'error': 'Token is required'}), 401
        
        payload = verify_jwt(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload['user_id']
        user = next((u for u in users.values() if u['user_id'] == user_id), None)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user_id': user['user_id'],
            'email': user['email'],
            'full_name': user['full_name'],
            'roles': user['roles'],
            'created_at': user['created_at']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'authentication',
        'timestamp': datetime.utcnow().isoformat(),
        'users_count': len(users)
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Authentication Service on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)
