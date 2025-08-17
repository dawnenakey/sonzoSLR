"""
Authentication API endpoints for SPOKHAND SIGNCUT
Provides login, registration, and user management endpoints
"""

from flask import Flask, request, jsonify, make_response
from functools import wraps
import asyncio
from auth_service import AuthService, UserService, UserRole, rate_limiter
import json

app = Flask(__name__)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'SPOKHAND SIGNCUT Authentication API',
        'version': '1.0.0'
    })

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        
        token = auth_header.split(' ')[1]
        payload = AuthService.verify_jwt(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user info to request context
        request.user = payload
        return f(*args, **kwargs)
    return decorated_function

def require_role(required_role):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'user'):
                return jsonify({'error': 'Authentication required'}), 401
            
            user_roles = request.user.get('roles', [])
            if not UserRole.has_permission(user_roles, required_role):
                return jsonify({'error': f'Insufficient permissions. Required role: {required_role}'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit(f):
    """Decorator to apply rate limiting"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get user ID from request (either from auth or IP for public endpoints)
        user_id = getattr(request, 'user', {}).get('user_id', request.remote_addr)
        
        if not rate_limiter.is_allowed(user_id):
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/auth/register', methods=['POST'])
@rate_limit
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['email', 'password', 'full_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate email format
        if '@' not in data['email']:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password strength
        if len(data['password']) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Create user
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            user = loop.run_until_complete(
                UserService.create_user(
                    email=data['email'],
                    password=data['password'],
                    full_name=data['full_name'],
                    roles=data.get('roles', [UserRole.TRANSLATOR])
                )
            )
            loop.close()
            
            return jsonify({
                'success': True,
                'message': 'User created successfully',
                'user': user
            }), 201
            
        except ValueError as e:
            loop.close()
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            loop.close()
            return jsonify({'error': f'Failed to create user: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
@rate_limit
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Authenticate user
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            user = loop.run_until_complete(
                UserService.authenticate_user(email, password)
            )
            loop.close()
            
            if not user:
                return jsonify({'error': 'Invalid email or password'}), 401
            
            # Generate JWT token
            token = AuthService.generate_jwt(
                user_id=user['id'],
                email=user['email'],
                roles=user['roles']
            )
            
            # Create response with token
            response = make_response(jsonify({
                'success': True,
                'message': 'Login successful',
                'token': token,
                'user': user
            }))
            
            # Set secure cookie (optional)
            response.set_cookie(
                'auth_token',
                token,
                httponly=True,
                secure=True,  # Set to False in development
                samesite='Strict',
                max_age=24*60*60  # 24 hours
            )
            
            return response, 200
            
        except Exception as e:
            loop.close()
            return jsonify({'error': f'Authentication failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/me', methods=['GET'])
@require_auth
@rate_limit
def get_current_user():
    """Get current user information"""
    try:
        user_id = request.user['user_id']
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            user = loop.run_until_complete(
                UserService.get_user_by_id(user_id)
            )
            loop.close()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            return jsonify({
                'success': True,
                'user': user
            }), 200
            
        except Exception as e:
            loop.close()
            return jsonify({'error': f'Failed to get user: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to get current user: {str(e)}'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout():
    """User logout endpoint"""
    try:
        # Log audit event
        AuthService.log_audit_event(
            user_id=request.user['user_id'],
            action='user_logout',
            resource='auth',
            details={'email': request.user['email']}
        )
        
        response = make_response(jsonify({
            'success': True,
            'message': 'Logout successful'
        }))
        
        # Clear auth cookie
        response.delete_cookie('auth_token')
        
        return response, 200
        
    except Exception as e:
        return jsonify({'error': f'Logout failed: {str(e)}'}), 500

@app.route('/api/admin/users', methods=['GET'])
@require_auth
@require_role(UserRole.ADMIN)
@rate_limit
def list_users():
    """List all users (admin only)"""
    try:
        admin_user_id = request.user['user_id']
        limit = request.args.get('limit', 50, type=int)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            users = loop.run_until_complete(
                UserService.list_users(admin_user_id, limit)
            )
            loop.close()
            
            return jsonify({
                'success': True,
                'users': users,
                'count': len(users)
            }), 200
            
        except Exception as e:
            loop.close()
            return jsonify({'error': f'Failed to list users: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to list users: {str(e)}'}), 500

@app.route('/api/admin/users/<user_id>/roles', methods=['PUT'])
@require_auth
@require_role(UserRole.ADMIN)
@rate_limit
def update_user_roles(user_id):
    """Update user roles (admin only)"""
    try:
        admin_user_id = request.user['user_id']
        data = request.get_json()
        
        if not data or 'roles' not in data:
            return jsonify({'error': 'Roles are required'}), 400
        
        new_roles = data['roles']
        if not isinstance(new_roles, list):
            return jsonify({'error': 'Roles must be a list'}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(
                UserService.update_user_roles(user_id, new_roles, admin_user_id)
            )
            loop.close()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'User roles updated successfully'
                }), 200
            else:
                return jsonify({'error': 'Failed to update user roles'}), 500
                
        except ValueError as e:
            loop.close()
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            loop.close()
            return jsonify({'error': f'Failed to update user roles: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to update user roles: {str(e)}'}), 500

@app.route('/api/auth/refresh', methods=['POST'])
@require_auth
@rate_limit
def refresh_token():
    """Refresh JWT token"""
    try:
        user_id = request.user['user_id']
        email = request.user['email']
        roles = request.user['roles']
        
        # Generate new token
        new_token = AuthService.generate_jwt(user_id, email, roles)
        
        # Log audit event
        AuthService.log_audit_event(
            user_id=user_id,
            action='token_refreshed',
            resource='auth',
            details={'email': email}
        )
        
        response = make_response(jsonify({
            'success': True,
            'token': new_token
        }))
        
        # Update auth cookie
        response.set_cookie(
            'auth_token',
            new_token,
            httponly=True,
            secure=True,
            samesite='Strict',
            max_age=24*60*60
        )
        
        return response, 200
        
    except Exception as e:
        return jsonify({'error': f'Token refresh failed: {str(e)}'}), 500

@app.route('/api/auth/validate', methods=['POST'])
@rate_limit
def validate_token():
    """Validate JWT token without requiring auth decorator"""
    try:
        data = request.get_json()
        if not data or 'token' not in data:
            return jsonify({'error': 'Token is required'}), 400
        
        token = data['token']
        payload = AuthService.verify_jwt(token)
        
        if payload:
            return jsonify({
                'success': True,
                'valid': True,
                'user': {
                    'user_id': payload['user_id'],
                    'email': payload['email'],
                    'roles': payload['roles']
                }
            }), 200
        else:
            return jsonify({
                'success': True,
                'valid': False
            }), 200
            
    except Exception as e:
        return jsonify({'error': f'Token validation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 