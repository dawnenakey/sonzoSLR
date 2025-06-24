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

    # Router
    try:
        http_method = event.get('httpMethod')
        # The actual path is available in requestContext
        path = event.get('requestContext', {}).get('resourcePath', event.get('path'))

        if http_method == 'OPTIONS':
            return create_response(200, 'CORS preflight OK', cors_headers)
        
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

        body = json.loads(event.get('body', '{}'))
        filename = body.get('filename')
        if not filename:
            return create_response(400, {'success': False, 'error': 'filename is required in the request body.'}, headers)

        content_type = body.get('contentType')
        if not content_type:
            return create_response(400, {'success': False, 'error': 'contentType is required in the request body.'}, headers)

        # Generate a unique key for the S3 object
        object_key = f"uploads/{session_id}/{filename}"

        # Generate the presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': object_key, 'ContentType': content_type},
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        logger.info(f"Generated presigned URL for session {session_id}")
        return create_response(200, {'success': True, 'uploadUrl': presigned_url, 'objectKey': object_key}, headers)

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