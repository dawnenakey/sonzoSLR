"""
Authentication Service for SPOKHAND SIGNCUT
Handles JWT authentication, user management, and role-based access control
"""

import jwt
import bcrypt
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import boto3
from botocore.exceptions import ClientError
import json
import os

# Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# DynamoDB setup
dynamodb = boto3.resource('dynamodb')
users_table = dynamodb.Table('spokhand-users')
audit_logs_table = dynamodb.Table('spokhand-audit-logs')

class UserRole:
    """User role definitions with cumulative permissions"""
    TRANSLATOR = 'translator'
    SEGMENTER = 'segmenter'
    QUALIFIER = 'qualifier'
    EXPERT = 'expert'
    ADMIN = 'admin'
    
    # Role hierarchy (cumulative permissions)
    ROLE_HIERARCHY = {
        TRANSLATOR: 1,
        SEGMENTER: 2,
        QUALIFIER: 3,
        EXPERT: 4,
        ADMIN: 5
    }
    
    @classmethod
    def has_permission(cls, user_roles: List[str], required_role: str) -> bool:
        """Check if user has required role or higher"""
        if not user_roles:
            return False
        
        required_level = cls.ROLE_HIERARCHY.get(required_role, 0)
        user_max_level = max(cls.ROLE_HIERARCHY.get(role, 0) for role in user_roles)
        return user_max_level >= required_level

class AuthService:
    """Main authentication service"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def generate_jwt(user_id: str, email: str, roles: List[str]) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'email': email,
            'roles': roles,
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_jwt(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def log_audit_event(user_id: str, action: str, resource: str, details: Dict[str, Any] = None):
        """Log audit event to DynamoDB"""
        try:
            audit_log = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'details': details or {},
                'timestamp': datetime.utcnow().isoformat(),
                'ip_address': details.get('ip_address', 'unknown') if details else 'unknown'
            }
            audit_logs_table.put_item(Item=audit_log)
        except Exception as e:
            print(f"Failed to log audit event: {e}")

class UserService:
    """User management service"""
    
    @staticmethod
    async def create_user(email: str, password: str, full_name: str, roles: List[str] = None) -> Dict[str, Any]:
        """Create new user"""
        try:
            # Check if user already exists
            existing_user = await UserService.get_user_by_email(email)
            if existing_user:
                raise ValueError("User already exists")
            
            # Validate roles
            if roles:
                for role in roles:
                    if role not in UserRole.ROLE_HIERARCHY:
                        raise ValueError(f"Invalid role: {role}")
            else:
                roles = [UserRole.TRANSLATOR]  # Default role
            
            # Create user
            user_id = str(uuid.uuid4())
            user = {
                'id': user_id,
                'email': email.lower(),
                'password_hash': AuthService.hash_password(password),
                'full_name': full_name,
                'roles': roles,
                'status': 'active',
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'last_login': None
            }
            
            users_table.put_item(Item=user)
            
            # Log audit event
            AuthService.log_audit_event(
                user_id=user_id,
                action='user_created',
                resource='user',
                details={'email': email, 'roles': roles}
            )
            
            # Return user without password
            user.pop('password_hash', None)
            return user
            
        except Exception as e:
            raise Exception(f"Failed to create user: {str(e)}")
    
    @staticmethod
    async def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password"""
        try:
            user = await UserService.get_user_by_email(email)
            if not user:
                return None
            
            if not AuthService.verify_password(password, user['password_hash']):
                return None
            
            # Update last login
            users_table.update_item(
                Key={'id': user['id']},
                UpdateExpression='SET last_login = :last_login',
                ExpressionAttributeValues={':last_login': datetime.utcnow().isoformat()}
            )
            
            # Log audit event
            AuthService.log_audit_event(
                user_id=user['id'],
                action='user_login',
                resource='auth',
                details={'email': email, 'ip_address': 'unknown'}
            )
            
            # Return user without password
            user.pop('password_hash', None)
            return user
            
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            response = users_table.scan(
                FilterExpression='email = :email',
                ExpressionAttributeValues={':email': email.lower()}
            )
            items = response.get('Items', [])
            return items[0] if items else None
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            response = users_table.get_item(Key={'id': user_id})
            user = response.get('Item')
            if user:
                user.pop('password_hash', None)
            return user
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None
    
    @staticmethod
    async def update_user_roles(user_id: str, new_roles: List[str], admin_user_id: str) -> bool:
        """Update user roles (admin only)"""
        try:
            # Verify admin permissions
            admin_user = await UserService.get_user_by_id(admin_user_id)
            if not admin_user or not UserRole.has_permission(admin_user['roles'], UserRole.ADMIN):
                raise ValueError("Insufficient permissions")
            
            # Validate new roles
            for role in new_roles:
                if role not in UserRole.ROLE_HIERARCHY:
                    raise ValueError(f"Invalid role: {role}")
            
            # Update user roles
            users_table.update_item(
                Key={'id': user_id},
                UpdateExpression='SET roles = :roles, updated_at = :updated_at',
                ExpressionAttributeValues={
                    ':roles': new_roles,
                    ':updated_at': datetime.utcnow().isoformat()
                }
            )
            
            # Log audit event
            AuthService.log_audit_event(
                user_id=admin_user_id,
                action='user_roles_updated',
                resource='user',
                details={'target_user_id': user_id, 'new_roles': new_roles}
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to update user roles: {str(e)}")
    
    @staticmethod
    async def list_users(admin_user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List all users (admin only)"""
        try:
            # Verify admin permissions
            admin_user = await UserService.get_user_by_id(admin_user_id)
            if not admin_user or not UserRole.has_permission(admin_user['roles'], UserRole.ADMIN):
                raise ValueError("Insufficient permissions")
            
            response = users_table.scan(Limit=limit)
            users = response.get('Items', [])
            
            # Remove password hashes
            for user in users:
                user.pop('password_hash', None)
            
            return users
            
        except Exception as e:
            raise Exception(f"Failed to list users: {str(e)}")

# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter (use Redis in production)"""
    
    def __init__(self):
        self.requests = {}
        self.max_requests = 100  # requests per hour
        self.window = 3600  # 1 hour in seconds
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        now = datetime.utcnow().timestamp()
        user_requests = self.requests.get(user_id, [])
        
        # Remove old requests outside window
        user_requests = [req_time for req_time in user_requests if now - req_time < self.window]
        
        if len(user_requests) < self.max_requests:
            user_requests.append(now)
            self.requests[user_id] = user_requests
            return True
        
        return False

# Global rate limiter instance
rate_limiter = RateLimiter() 