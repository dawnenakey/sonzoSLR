import json
import os
import boto3
import logging
import decimal
import uuid
from datetime import datetime

# --- Configuration ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
VERSION = "v1.7" # Deployment version tracker

# --- AWS Clients ---
DYNAMODB_TABLE_NAME = os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE_NAME)
S3_BUCKET_NAME = os.environ.get('S3_BUCKET', 'spokhand-data')
s3_client = boto3.client('s3')

# --- Helper Classes ---
class DecimalEncoder(json.JSONEncoder):
    """Helper class to convert a DynamoDB item to JSON."""
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)

# --- Main Handler ---
def lambda_handler(event, context):
    """
    Main Lambda handler for API Gateway requests.
    Routes requests to the appropriate handler based on HTTP method and path.
    """
    logger.info(f"--- Spokhand Lambda v{VERSION} ---")
    logger.info(f"Received event: {json.dumps(event)}")

    # Standard CORS headers
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
    }

    try:
        http_method = event.get('httpMethod')
        path = event.get('requestContext', {}).get('resourcePath', event.get('path'))
        path_params = event.get('pathParameters', {})
        logger.info(f"path: {path}")
        logger.info(f"path_params: {path_params}")

        if http_method == 'OPTIONS':
            return create_response(200, 'CORS preflight OK', cors_headers)

        # Route based on presence of 'proxy' path parameter
        if 'proxy' in path_params:
            proxy = path_params['proxy']
            parts = proxy.split('/')
            if len(parts) >= 2:
                video_id = '/'.join(parts[:-1])
                action = parts[-1]
                if action == 'annotations':
                    if http_method == 'GET':
                        return handle_get_annotations(event, cors_headers, video_id)
                    elif http_method == 'POST':
                        return handle_create_annotation(event, cors_headers, video_id)
                elif action == 'stream':
                    if http_method == 'GET':
                        return handle_get_stream(event, cors_headers, video_id)
                else:
                    return create_response(404, {'success': False, 'error': 'Unknown action'}, cors_headers)
            else:
                return create_response(400, {'success': False, 'error': 'Invalid proxy path'}, cors_headers)

        # Other routes
        elif http_method == 'POST' and path == '/sessions':
            return handle_create_session(event, cors_headers)
        elif http_method == 'POST' and path == '/sessions/{sessionId}/upload-video':
            return handle_generate_upload_url(event, cors_headers)
        elif http_method == 'GET' and path == '/sessions':
            return handle_get_sessions(event, cors_headers)
        else:
            logger.warning(f"No route matched for method '{http_method}' and path '{path}'")
            return create_response(404, {'success': False, 'error': 'Not Found'}, cors_headers)

    except Exception as e:
        logger.error(f"!!! Unhandled exception: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Internal Server Error'}, cors_headers)


# --- Route Handlers ---
def handle_create_session(event, headers):
    """Handles POST /sessions - Creates a new session."""
    try:
        body = json.loads(event.get('body', '{}'))
        session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        item = {
            'timestamp': timestamp,  # DynamoDB primary key
            'session_id': session_id,
            'name': body.get('name', 'Unnamed Session'),
            'description': body.get('description', ''),
            'createdAt': timestamp
        }
        
        table.put_item(Item=item)
        logger.info(f"Successfully created session {session_id}")
        return create_response(201, {'success': True, 'session_id': session_id}, headers)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {str(e)}")
        return create_response(400, {'success': False, 'error': 'Invalid JSON format'}, headers)
    except Exception as e:
        logger.error(f"Error creating session in DynamoDB: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Could not create session'}, headers)


def handle_generate_upload_url(event, headers):
    """Handles POST /sessions/{sessionId}/upload-video - Generates a presigned S3 URL."""
    try:
        session_id = event.get('pathParameters', {}).get('sessionId')
        if not session_id:
            return create_response(400, {'success': False, 'error': 'sessionId path parameter is required.'}, headers)

        # Handle both JSON and FormData requests
        body = event.get('body', '{}')
        content_type = event.get('headers', {}).get('content-type', '')
        
        filename = None
        file_content_type = None
        
        if 'application/json' in content_type:
            # JSON request
            try:
                body_data = json.loads(body) if body else {}
                filename = body_data.get('filename')
                file_content_type = body_data.get('contentType')
            except json.JSONDecodeError:
                return create_response(400, {'success': False, 'error': 'Invalid JSON in request body.'}, headers)
        else:
            # FormData request - extract from multipart form data
            # For now, we'll use default values since FormData parsing in Lambda is complex
            # The frontend should send the filename and content type in the request
            filename = f"video_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp4"
            file_content_type = 'video/mp4'
            
            # Try to extract from body if it's a simple string
            if body and isinstance(body, str):
                # Look for filename in the body
                import re
                filename_match = re.search(r'filename="([^"]+)"', body)
                if filename_match:
                    filename = filename_match.group(1)
                
                # Look for content type
                content_type_match = re.search(r'Content-Type: ([^\r\n]+)', body)
                if content_type_match:
                    file_content_type = content_type_match.group(1).strip()

        if not filename:
            return create_response(400, {'success': False, 'error': 'filename is required in the request body.'}, headers)

        if not file_content_type:
            file_content_type = 'video/mp4'  # Default content type

        # Generate a unique key for the S3 object
        object_key = f"uploads/{session_id}/{filename}"

        # Generate the presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': object_key, 'ContentType': file_content_type},
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        # Return a video object that matches what the frontend expects
        video_object = {
            'id': object_key,  # Use object_key as the video ID
            'name': filename,
            'url': f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{object_key}",
            'sessionId': session_id,
            'uploadUrl': presigned_url
        }
        
        logger.info(f"Generated presigned URL for session {session_id}, filename: {filename}")
        return create_response(200, {'success': True, 'video': video_object, 'uploadUrl': presigned_url, 'objectKey': object_key}, headers)

    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Could not generate upload URL'}, headers)


def handle_get_sessions(event, headers):
    """Handles GET /sessions - Retrieves all sessions."""
    try:
        response = table.scan()
        items = response.get('Items', [])
        logger.info(f"Found {len(items)} sessions.")
        return create_response(200, {'success': True, 'sessions': items}, headers)

    except Exception as e:
        logger.error(f"Error scanning sessions from DynamoDB: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Could not retrieve sessions'}, headers)


def handle_get_annotations(event, headers, video_id):
    """Handles GET /videos/{proxy+}/annotations - Retrieves all annotations for a video (videoId may contain slashes)."""
    try:
        if not video_id:
            return create_response(400, {'success': False, 'error': 'videoId (proxy path parameter) is required.'}, headers)
        response = table.scan(
            FilterExpression='video_id = :v',
            ExpressionAttributeValues={':v': video_id}
        )
        items = response.get('Items', [])
        return create_response(200, {'success': True, 'annotations': items}, headers)
    except Exception as e:
        logger.error(f"Error retrieving annotations for video {video_id}: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Could not retrieve annotations'}, headers)


def handle_create_annotation(event, headers, video_id):
    """Handles POST /videos/{proxy+}/annotations - Creates a new annotation for a video (videoId may contain slashes)."""
    try:
        if not video_id:
            return create_response(400, {'success': False, 'error': 'videoId (proxy path parameter) is required.'}, headers)
        body = json.loads(event.get('body', '{}'))
        timestamp = datetime.utcnow().isoformat()
        item = {
            'timestamp': timestamp,
            'video_id': video_id,
            'annotation': body.get('annotation', ''),
            'createdAt': timestamp
        }
        table.put_item(Item=item)
        logger.info(f"Successfully created annotation for video {video_id}")
        return create_response(201, {'success': True, 'video_id': video_id}, headers)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {str(e)}")
        return create_response(400, {'success': False, 'error': 'Invalid JSON format'}, headers)
    except Exception as e:
        logger.error(f"Error creating annotation in DynamoDB: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Could not create annotation'}, headers)


# Placeholder for stream handler
def handle_get_stream(event, headers, video_id):
    return create_response(501, {'success': False, 'error': 'Stream not implemented'}, headers)


# --- Utility Functions ---
def create_response(status_code, body, headers={}):
    """Creates a valid API Gateway proxy response."""
    # Ensure body is a JSON string
    if not isinstance(body, str):
        body = json.dumps(body, cls=DecimalEncoder)
        
    response = {
        'statusCode': status_code,
        'headers': headers,
        'body': body
    }
    logger.info(f"Returning response: {json.dumps(response, cls=DecimalEncoder)}")
    return response 