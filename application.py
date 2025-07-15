import os
import sys
from pathlib import Path
from flask import Flask, jsonify, request
from datetime import datetime

# Add the application directory to the Python path
application_path = Path(__file__).parent
sys.path.insert(0, str(application_path))

# Create Flask app for health checks and API endpoints
app = Flask(__name__)

# Import job coaching service
try:
    from src.job_coaching_service import JobCoachingService, MOCK_SESSIONS, MOCK_RESOURCES
    job_coaching = JobCoachingService()
except ImportError:
    # Fallback for development
    job_coaching = None

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

# Job Coaching API endpoints
@app.route('/api/job-coaching/sessions', methods=['GET'])
def get_sessions():
    """Get user sessions"""
    user_id = request.args.get('user_id', 'user123')  # Default for demo
    
    if job_coaching:
        sessions = job_coaching.get_user_sessions(user_id)
    else:
        sessions = MOCK_SESSIONS
    
    return jsonify(sessions), 200

@app.route('/api/job-coaching/sessions', methods=['POST'])
def create_session():
    """Create a new coaching session"""
    data = request.get_json()
    user_id = data.get('user_id', 'user123')
    session_type = data.get('session_type', 'interview')
    title = data.get('title')
    
    if job_coaching:
        session = job_coaching.create_session(user_id, session_type, title)
    else:
        session = {
            'session_id': 'mock-session-1',
            'user_id': user_id,
            'session_type': session_type,
            'title': title or f"{session_type.title()} Practice Session",
            'status': 'active',
            'created_at': datetime.utcnow().isoformat()
        }
    
    return jsonify(session), 201

@app.route('/api/job-coaching/sessions/<session_id>/feedback', methods=['POST'])
def generate_feedback(session_id):
    """Generate feedback for a session"""
    if job_coaching:
        # In real implementation, you'd get video data from request
        feedback = job_coaching.generate_feedback(session_id, b'mock_video_data')
    else:
        feedback = {
            'confidence': 85,
            'clarity': 90,
            'body_language': 88,
            'overall_score': 88,
            'suggestions': [
                'Great eye contact maintained throughout',
                'Consider slowing down slightly for complex phrases',
                'Excellent use of facial expressions'
            ]
        }
    
    return jsonify(feedback), 200

@app.route('/api/job-coaching/progress/<user_id>')
def get_user_progress(user_id):
    """Get user progress statistics"""
    if job_coaching:
        progress = job_coaching.get_user_progress(user_id)
    else:
        progress = {
            'total_sessions': 12,
            'completed_sessions': 8,
            'average_score': 87,
            'total_hours': 4.0,
            'skills_mastered': 6
        }
    
    return jsonify(progress), 200

@app.route('/api/job-coaching/resources')
def get_resources():
    """Get career resources"""
    category = request.args.get('category')
    
    if job_coaching:
        resources = job_coaching.get_career_resources(category)
    else:
        resources = MOCK_RESOURCES
    
    return jsonify(resources), 200

# Create WSGI application
application = app 