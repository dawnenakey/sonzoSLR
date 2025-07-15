import os
import sys
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

# Add the application directory to the Python path
application_path = Path(__file__).parent
sys.path.insert(0, str(application_path))

# Create Flask app for health checks and API endpoints
app = Flask(__name__)
CORS(app, origins=['*'], methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

@app.route('/api/health')
def health_check():
    """Health check endpoint for connection monitoring"""
    try:
        # Basic health check - you can add more sophisticated checks here
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/')
def index():
    """Main application endpoint"""
    return jsonify({
        'message': 'SpokHand SLR API',
        'version': '1.0.0',
        'status': 'running'
    }), 200

# Create WSGI application
application = app 