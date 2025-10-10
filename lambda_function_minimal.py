import json
import boto3
import os
import logging
from datetime import datetime

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Your existing Lambda function version
VERSION = "2.1.0"

def create_response(status_code, body, headers=None):
    """Create a standardized API Gateway response"""
    if headers is None:
        headers = {}
    
    response = {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET,PUT,DELETE'
        }
    }
    
    # Add custom headers
    response['headers'].update(headers)
    
    # Convert body to JSON if it's not already a string
    if isinstance(body, (dict, list)):
        response['body'] = json.dumps(body)
    else:
        response['body'] = str(body)
    
    return response

def lambda_handler(event, context):
    """
    Main Lambda handler for API Gateway requests.
    Routes requests to the appropriate handler based on HTTP method and path.
    """
    logger.info(f"--- SpokHand Lambda v{VERSION} ---")
    logger.info(f"Received event: {json.dumps(event)}")
    
    # CORS headers
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET,PUT,DELETE'
    }
    
    try:
        # Handle CORS preflight requests
        if event.get('httpMethod') == 'OPTIONS':
            return create_response(200, {}, cors_headers)
        
        # Extract request details
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        path_params = event.get('pathParameters') or {}
        query_params = event.get('queryStringParameters') or {}
        body = event.get('body', '{}')
        
        # Parse JSON body if it's a string
        if isinstance(body, str):
            try:
                body = json.loads(body) if body else {}
            except json.JSONDecodeError:
                body = {}
        
        logger.info(f"Processing {http_method} {path}")
        
        # Route to appropriate handler
        if path.startswith('/api/asl-lex'):
            # ASL-LEX endpoints (simplified - return not implemented)
            return create_response(501, {
                'success': False,
                'error': 'ASL-LEX endpoints not available in this deployment'
            }, cors_headers)
        else:
            # Handle main endpoints
            return handle_main_endpoints(http_method, path, path_params, query_params, body, cors_headers)
            
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return create_response(500, {
            'success': False,
            'error': 'Internal server error'
        }, cors_headers)

def handle_main_endpoints(method, path, path_params, query_params, body, cors_headers):
    """Handle main application endpoints"""
    try:
        # Route based on path and method
        if method == 'POST' and path == '/sessions':
            return handle_create_session(body, cors_headers)
        elif method == 'POST' and path.startswith('/sessions/') and path.endswith('/upload-video'):
            return handle_generate_upload_url(path_params, cors_headers)
        elif method == 'GET' and path == '/sessions':
            return handle_get_sessions(cors_headers)
        elif method == 'GET' and path == '/videos':
            return handle_get_videos(cors_headers)
        elif method == 'GET' and path.startswith('/videos/') and not path.endswith('/annotations') and not path.endswith('/stream'):
            return handle_get_video(path_params, cors_headers)
        elif method == 'GET' and path.endswith('/annotations'):
            return handle_get_annotations(path_params, cors_headers)
        elif method == 'POST' and path.endswith('/annotations'):
            return handle_create_annotation(path_params, body, cors_headers)
        else:
            return create_response(404, {'success': False, 'error': 'Not Found'}, cors_headers)
            
    except Exception as e:
        logger.error(f"Error in main endpoints handler: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Internal Server Error'}, cors_headers)

def handle_create_session(body, cors_headers):
    """Create a new session"""
    try:
        # Initialize DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection'))
        
        # Generate session ID
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.utcnow().isoformat()
        
        # Create session item
        session_item = {
            'session_id': session_id,
            'timestamp': 'session',  # Special timestamp for session records
            'name': body.get('name', f'Session {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}'),
            'description': body.get('description', ''),
            'status': 'active',
            'createdAt': timestamp,
            'updatedAt': timestamp
        }
        
        table.put_item(Item=session_item)
        
        logger.info(f"Created session {session_id}")
        
        return create_response(201, {
            'success': True,
            'session_id': session_id,
            'session': {
                'id': session_id,
                'name': session_item['name'],
                'description': session_item['description'],
                'status': 'active',
                'createdAt': timestamp,
                'updatedAt': timestamp
            }
        }, cors_headers)
        
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return create_response(500, {
            'success': False,
            'error': 'Failed to create session'
        }, cors_headers)

def handle_generate_upload_url(path_params, cors_headers):
    """Generate presigned URL for video upload and store metadata"""
    try:
        # Extract session_id from path
        session_id = path_params.get('sessionId')  # API Gateway uses 'sessionId' not 'session_id'
        if not session_id:
            return create_response(400, {'success': False, 'error': 'Session ID is required'}, cors_headers)
        
        # Initialize AWS services
        s3_client = boto3.client('s3')
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection'))
        
        # Generate unique video ID
        video_id = f"video_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{session_id}"
        
        # Generate presigned URL for S3 upload
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': os.environ.get('S3_BUCKET_NAME', 'spokhand-data'),
                'Key': f"videos/{session_id}/{video_id}.mp4",
                'ContentType': 'video/mp4'
            },
            ExpiresIn=3600  # 1 hour
        )
        
        # Store video metadata in DynamoDB
        timestamp = datetime.utcnow().isoformat()
        video_item = {
            'session_id': session_id,
            'timestamp': timestamp,
            'video_id': video_id,
            'filename': f"{video_id}.mp4",
            'status': 'uploading',
            'createdAt': timestamp,
            'size': 0,  # Will be updated after upload
            'duration': 0  # Will be updated after processing
        }
        
        table.put_item(Item=video_item)
        
        logger.info(f"Generated upload URL for video {video_id} in session {session_id}")
        
        return create_response(200, {
            'success': True,
            'uploadUrl': presigned_url,
            'video': {
                'id': video_id,
                'sessionId': session_id,
                'filename': f"{video_id}.mp4",
                'status': 'uploading',
                'uploadedAt': timestamp
            }
        }, cors_headers)
        
    except Exception as e:
        logger.error(f"Error generating upload URL: {str(e)}")
        return create_response(500, {
            'success': False,
            'error': 'Failed to generate upload URL'
        }, cors_headers)

def handle_get_videos(cors_headers):
    """Handle GET /videos - List all videos"""
    try:
        # Initialize DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection'))
        
        # Scan the table for all video records
        response = table.scan(
            FilterExpression='attribute_exists(video_id)'
        )
        
        videos = []
        for item in response.get('Items', []):
            # Only include items that have video information
            if 'video_id' in item and 'filename' in item:
                video = {
                    'id': item.get('video_id', ''),
                    'sessionId': item.get('session_id', ''),
                    'filename': item.get('filename', ''),
                    'size': item.get('size', 0),
                    'duration': item.get('duration', 0),
                    'uploadedAt': item.get('createdAt', item.get('timestamp', '')),
                    'status': item.get('status', 'ready'),
                    'url': f"https://{os.environ.get('S3_BUCKET_NAME', 'spokhand-data')}.s3.amazonaws.com/videos/{item.get('session_id', '')}/{item.get('video_id', '')}.mp4"
                }
                videos.append(video)
        
        logger.info(f"Found {len(videos)} videos")
        return create_response(200, {
            'success': True,
            'videos': videos,
            'total': len(videos)
        }, cors_headers)
        
    except Exception as e:
        logger.error(f"Error getting videos: {str(e)}")
        return create_response(500, {
            'success': False,
            'error': 'Failed to retrieve videos'
        }, cors_headers)

def handle_get_video(path_params, cors_headers):
    """Handle GET /videos/{video_id} - Get specific video"""
    try:
        video_id = path_params.get('videoId')  # API Gateway might use 'videoId' instead of 'video_id'
        if not video_id:
            return create_response(400, {
                'success': False,
                'error': 'Video ID is required'
            }, cors_headers)
        
        # Initialize DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection'))
        
        # Scan for the specific video
        response = table.scan(
            FilterExpression='video_id = :video_id',
            ExpressionAttributeValues={':video_id': video_id}
        )
        
        items = response.get('Items', [])
        if not items:
            return create_response(404, {
                'success': False,
                'error': 'Video not found'
            }, cors_headers)
        
        item = items[0]  # Take the first match
        video = {
            'id': item.get('video_id', ''),
            'sessionId': item.get('session_id', ''),
            'filename': item.get('filename', ''),
            'size': item.get('size', 0),
            'duration': item.get('duration', 0),
            'uploadedAt': item.get('createdAt', item.get('timestamp', '')),
            'status': item.get('status', 'ready'),
            'url': f"https://{os.environ.get('S3_BUCKET_NAME', 'spokhand-data')}.s3.amazonaws.com/videos/{item.get('session_id', '')}/{item.get('video_id', '')}.mp4"
        }
        
        return create_response(200, {
            'success': True,
            'video': video
        }, cors_headers)
        
    except Exception as e:
        logger.error(f"Error getting video: {str(e)}")
        return create_response(500, {
            'success': False,
            'error': 'Failed to retrieve video'
        }, cors_headers)

def handle_get_sessions(cors_headers):
    """Get all sessions"""
    try:
        # Initialize DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection'))
        
        # Scan for session records
        response = table.scan(
            FilterExpression='timestamp = :session_timestamp',
            ExpressionAttributeValues={':session_timestamp': 'session'}
        )
        
        sessions = []
        for item in response.get('Items', []):
            session = {
                'id': item.get('session_id', ''),
                'name': item.get('name', ''),
                'description': item.get('description', ''),
                'status': item.get('status', 'active'),
                'createdAt': item.get('createdAt', ''),
                'updatedAt': item.get('updatedAt', '')
            }
            sessions.append(session)
        
        return create_response(200, {
            'success': True,
            'sessions': sessions
        }, cors_headers)
        
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        return create_response(500, {
            'success': False,
            'error': 'Failed to retrieve sessions'
        }, cors_headers)

def handle_get_annotations(path_params, cors_headers):
    """Get annotations for a video"""
    return create_response(200, {
        'success': True,
        'annotations': []
    }, cors_headers)

def handle_create_annotation(path_params, body, cors_headers):
    """Create a new annotation"""
    return create_response(200, {
        'success': True,
        'message': 'Annotation created'
    }, cors_headers)
