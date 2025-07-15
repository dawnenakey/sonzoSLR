import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

class JobCoachingService:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.s3 = boto3.client('s3')
        self.sessions_table = self.dynamodb.Table('job_coaching_sessions')
        self.resources_table = self.dynamodb.Table('job_coaching_resources')
        
    def create_session(self, user_id: str, session_type: str, title: str = None) -> Dict:
        """Create a new job coaching session"""
        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'session_type': session_type,
            'title': title or f"{session_type.title()} Practice Session",
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'video_url': None,
            'feedback': None,
            'metrics': {
                'confidence': 0,
                'clarity': 0,
                'body_language': 0,
                'overall_score': 0
            },
            'duration': 0,
            'suggestions': []
        }
        
        try:
            self.sessions_table.put_item(Item=session_data)
            return session_data
        except ClientError as e:
            print(f"Error creating session: {e}")
            return None
    
    def update_session_video(self, session_id: str, video_url: str) -> bool:
        """Update session with video recording"""
        try:
            self.sessions_table.update_item(
                Key={'session_id': session_id},
                UpdateExpression='SET video_url = :url, updated_at = :time',
                ExpressionAttributeValues={
                    ':url': video_url,
                    ':time': datetime.utcnow().isoformat()
                }
            )
            return True
        except ClientError as e:
            print(f"Error updating session video: {e}")
            return False
    
    def generate_feedback(self, session_id: str, video_data: bytes) -> Dict:
        """Generate AI-powered feedback for the session"""
        # TODO: Integrate with your existing sign spotting service
        # For now, return mock feedback
        
        # Simulate AI analysis
        import random
        confidence = random.randint(70, 95)
        clarity = random.randint(75, 90)
        body_language = random.randint(80, 95)
        overall_score = (confidence + clarity + body_language) // 3
        
        suggestions = [
            "Great eye contact maintained throughout the session",
            "Consider slowing down slightly for complex phrases",
            "Excellent use of facial expressions",
            "Try to maintain consistent signing space",
            "Good use of body language to emphasize key points"
        ]
        
        feedback_data = {
            'confidence': confidence,
            'clarity': clarity,
            'body_language': body_language,
            'overall_score': overall_score,
            'suggestions': suggestions[:3],  # Top 3 suggestions
            'generated_at': datetime.utcnow().isoformat()
        }
        
        try:
            self.sessions_table.update_item(
                Key={'session_id': session_id},
                UpdateExpression='SET feedback = :feedback, metrics = :metrics, updated_at = :time',
                ExpressionAttributeValues={
                    ':feedback': feedback_data,
                    ':metrics': {
                        'confidence': confidence,
                        'clarity': clarity,
                        'body_language': body_language,
                        'overall_score': overall_score
                    },
                    ':time': datetime.utcnow().isoformat()
                }
            )
            return feedback_data
        except ClientError as e:
            print(f"Error updating feedback: {e}")
            return None
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user"""
        try:
            response = self.sessions_table.query(
                IndexName='user_id-index',
                KeyConditionExpression='user_id = :user_id',
                ExpressionAttributeValues={':user_id': user_id}
            )
            return response.get('Items', [])
        except ClientError as e:
            print(f"Error getting user sessions: {e}")
            return []
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session by ID"""
        try:
            response = self.sessions_table.get_item(Key={'session_id': session_id})
            return response.get('Item')
        except ClientError as e:
            print(f"Error getting session: {e}")
            return None
    
    def complete_session(self, session_id: str) -> bool:
        """Mark session as completed"""
        try:
            self.sessions_table.update_item(
                Key={'session_id': session_id},
                UpdateExpression='SET status = :status, updated_at = :time',
                ExpressionAttributeValues={
                    ':status': 'completed',
                    ':time': datetime.utcnow().isoformat()
                }
            )
            return True
        except ClientError as e:
            print(f"Error completing session: {e}")
            return False
    
    def get_user_progress(self, user_id: str) -> Dict:
        """Get user's overall progress and statistics"""
        sessions = self.get_user_sessions(user_id)
        
        if not sessions:
            return {
                'total_sessions': 0,
                'completed_sessions': 0,
                'average_score': 0,
                'total_hours': 0,
                'skills_mastered': 0
            }
        
        completed_sessions = [s for s in sessions if s.get('status') == 'completed']
        total_sessions = len(sessions)
        completed_count = len(completed_sessions)
        
        if completed_count == 0:
            return {
                'total_sessions': total_sessions,
                'completed_sessions': completed_count,
                'average_score': 0,
                'total_hours': 0,
                'skills_mastered': 0
            }
        
        # Calculate average score
        total_score = sum(s.get('metrics', {}).get('overall_score', 0) for s in completed_sessions)
        average_score = total_score // completed_count if completed_count > 0 else 0
        
        # Calculate total hours (assuming 30 minutes per session)
        total_hours = (completed_count * 0.5)
        
        # Count skills mastered (sessions with score > 85)
        skills_mastered = len([s for s in completed_sessions 
                             if s.get('metrics', {}).get('overall_score', 0) > 85])
        
        return {
            'total_sessions': total_sessions,
            'completed_sessions': completed_count,
            'average_score': average_score,
            'total_hours': round(total_hours, 1),
            'skills_mastered': skills_mastered
        }
    
    def get_career_resources(self, category: str = None) -> List[Dict]:
        """Get career resources and tutorials"""
        try:
            if category:
                response = self.resources_table.query(
                    IndexName='category-index',
                    KeyConditionExpression='category = :category',
                    ExpressionAttributeValues={':category': category}
                )
            else:
                response = self.resources_table.scan()
            
            return response.get('Items', [])
        except ClientError as e:
            print(f"Error getting career resources: {e}")
            return []
    
    def create_career_resource(self, title: str, category: str, video_url: str, 
                              transcript: str, difficulty: str = 'beginner') -> Dict:
        """Create a new career resource"""
        resource_id = str(uuid.uuid4())
        resource_data = {
            'resource_id': resource_id,
            'title': title,
            'category': category,
            'video_url': video_url,
            'transcript': transcript,
            'difficulty': difficulty,
            'created_at': datetime.utcnow().isoformat(),
            'views': 0,
            'rating': 0
        }
        
        try:
            self.resources_table.put_item(Item=resource_data)
            return resource_data
        except ClientError as e:
            print(f"Error creating resource: {e}")
            return None

# Mock data for development
MOCK_SESSIONS = [
    {
        'session_id': '1',
        'user_id': 'user123',
        'session_type': 'interview',
        'title': 'Interview Practice - Tell me about yourself',
        'status': 'completed',
        'created_at': '2024-01-15T10:00:00',
        'metrics': {
            'confidence': 85,
            'clarity': 90,
            'body_language': 88,
            'overall_score': 88
        },
        'suggestions': [
            'Great eye contact maintained throughout',
            'Consider slowing down slightly for complex phrases',
            'Excellent use of facial expressions'
        ]
    },
    {
        'session_id': '2',
        'user_id': 'user123',
        'session_type': 'resume',
        'title': 'Resume Workshop - Experience Section',
        'status': 'completed',
        'created_at': '2024-01-16T14:30:00',
        'metrics': {
            'confidence': 92,
            'clarity': 88,
            'body_language': 85,
            'overall_score': 88
        },
        'suggestions': [
            'Strong presentation of key achievements',
            'Good use of professional vocabulary',
            'Consider adding more specific examples'
        ]
    }
]

MOCK_RESOURCES = [
    {
        'resource_id': '1',
        'title': 'Common Interview Questions in ASL',
        'category': 'interview',
        'video_url': 'https://example.com/interview-questions.mp4',
        'transcript': 'Learn how to answer common interview questions in ASL...',
        'difficulty': 'beginner',
        'views': 150,
        'rating': 4.5
    },
    {
        'resource_id': '2',
        'title': 'Professional Networking in ASL',
        'category': 'networking',
        'video_url': 'https://example.com/networking.mp4',
        'transcript': 'Master the art of networking in ASL...',
        'difficulty': 'intermediate',
        'views': 89,
        'rating': 4.2
    }
] 