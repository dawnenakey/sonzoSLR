import json
import boto3
import os
import logging
from datetime import datetime
from asl_lex_service import ASLLexDataManager

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize the ASL-LEX service
asl_lex_service = ASLLexDataManager()

# Your existing Lambda function version
VERSION = "2.0.0"

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
    logger.info(f"--- Spokhand Lambda v{VERSION} ---")
    logger.info(f"Received event: {json.dumps(event)}")

    # Standard CORS headers
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET,PUT,DELETE'
    }

    try:
        http_method = event.get('httpMethod')
        path = event.get('requestContext', {}).get('resourcePath', event.get('path', ''))
        path_parameters = event.get('pathParameters', {})
        query_string_parameters = event.get('queryStringParameters', {})
        body = event.get('body', '{}')
        
        # Parse body if it's a string
        if isinstance(body, str):
            try:
                body = json.loads(body) if body else {}
            except json.JSONDecodeError:
                body = {}
        
        logger.info(f"HTTP Method: {http_method}, Path: {path}")

        # Handle CORS preflight
        if http_method == 'OPTIONS':
            return create_response(200, 'CORS preflight OK', cors_headers)

        # Route ASL-LEX endpoints
        if path.startswith('/api/asl-lex'):
            return handle_asl_lex_endpoints(http_method, path, path_parameters, query_string_parameters, body, cors_headers)
        
        # Route existing endpoints (your current functionality)
        return handle_existing_endpoints(http_method, path, path_parameters, body, cors_headers)

    except Exception as e:
        logger.error(f"!!! Unhandled exception: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Internal Server Error'}, cors_headers)

def handle_asl_lex_endpoints(method, path, path_params, query_params, body, cors_headers):
    """Handle ASL-LEX specific endpoints"""
    try:
        # Parse the path to determine the endpoint
        path_parts = path.strip('/').split('/')
        
        # Remove the /api/asl-lex prefix
        if len(path_parts) >= 3 and path_parts[0] == 'api' and path_parts[1] == 'asl-lex':
            endpoint = '/'.join(path_parts[2:])
        else:
            endpoint = '/'.join(path_parts)
        
        logger.info(f"Routing ASL-LEX request: {method} {endpoint}")
        
        # Route to appropriate ASL-LEX handler
        if endpoint == 'signs' and method == 'GET':
            return handle_get_signs(query_params, cors_headers)
        elif endpoint == 'signs' and method == 'POST':
            return handle_create_sign(body, cors_headers)
        elif endpoint == 'sign-types' and method == 'GET':
            return handle_get_sign_types(cors_headers)
        elif endpoint == 'sign-types' and method == 'POST':
            return handle_add_custom_sign_type(body, cors_headers)
        elif endpoint == 'sign-types/custom' and method == 'GET':
            return handle_get_custom_sign_types(cors_headers)
        elif endpoint == 'signs/batch-update-type' and method == 'POST':
            return handle_batch_update_sign_types(body, cors_headers)
        elif endpoint == 'validate-asl-sign' and method == 'POST':
            return handle_validate_asl_sign(body, cors_headers)
        elif endpoint == 'analytics/sign-types' and method == 'GET':
            return handle_get_sign_type_analytics(cors_headers)
        elif endpoint == 'bulk-upload' and method == 'POST':
            return handle_bulk_upload(body, cors_headers)
        elif endpoint == 'bulk-upload/jobs' and method == 'GET':
            return handle_get_bulk_upload_jobs(cors_headers)
        elif endpoint == 'bulk-upload/template' and method == 'GET':
            return handle_get_bulk_upload_template(cors_headers)
        elif endpoint == 'statistics' and method == 'GET':
            return handle_get_statistics(cors_headers)
        elif endpoint == 'bulk-upload/jobs/{jobId}' and method == 'GET':
            job_id = path_params.get('jobId')
            return handle_get_bulk_upload_job(job_id, cors_headers)
        elif endpoint == 'bulk-upload/jobs/{jobId}/cancel' and method == 'POST':
            job_id = path_params.get('jobId')
            return handle_cancel_bulk_upload_job(job_id, cors_headers)
        elif endpoint == 'upload-video-with-metadata' and method == 'POST':
            return handle_upload_video_with_metadata(body, cors_headers)
        else:
            return create_response(404, {'error': f'ASL-LEX endpoint not found: {method} {endpoint}'}, cors_headers)
            
    except Exception as e:
        logger.error(f"Error in ASL-LEX handler: {str(e)}")
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_existing_endpoints(method, path, path_params, body, cors_headers):
    """Handle your existing endpoints"""
    try:
        # Your existing routing logic here
        if method == 'POST' and path == '/sessions':
            return handle_create_session(body, cors_headers)
        elif method == 'POST' and path.startswith('/sessions/') and path.endswith('/upload-video'):
            return handle_generate_upload_url(path_params, cors_headers)
        elif method == 'GET' and path == '/sessions':
            return handle_get_sessions(cors_headers)
        elif method == 'GET' and path.endswith('/annotations'):
            return handle_get_annotations(path_params, cors_headers)
        elif method == 'POST' and path.endswith('/annotations'):
            return handle_create_annotation(path_params, body, cors_headers)
        else:
            return create_response(404, {'success': False, 'error': 'Not Found'}, cors_headers)
            
    except Exception as e:
        logger.error(f"Error in existing handler: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Internal Server Error'}, cors_headers)

# ASL-LEX Handler Functions
def handle_get_signs(query_params, cors_headers):
    """Handle GET /api/asl-lex/signs"""
    try:
        filters = {}
        if query_params:
            if 'status' in query_params and query_params['status'] != 'all':
                filters['status'] = query_params['status']
            if 'handshape' in query_params and query_params['handshape'] != 'all':
                filters['handshape'] = query_params['handshape']
            if 'location' in query_params and query_params['location'] != 'all':
                filters['location'] = query_params['location']
            if 'sign_type' in query_params and query_params['sign_type'] != 'all':
                filters['sign_type'] = query_params['sign_type']
            if 'search' in query_params:
                filters['search'] = query_params['search']
        
        signs = asl_lex_service.list_signs(**filters)
        return create_response(200, [sign.__dict__ for sign in signs], cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_create_sign(body, cors_headers):
    """Handle POST /api/asl-lex/signs"""
    try:
        sign_id = asl_lex_service.create_sign(body)
        return create_response(201, {'id': sign_id, 'message': 'Sign created successfully'}, cors_headers)
    except Exception as e:
        return create_response(400, {'error': str(e)}, cors_headers)

def handle_get_sign_types(cors_headers):
    """Handle GET /api/asl-lex/sign-types"""
    try:
        sign_types = asl_lex_service.get_sign_types()
        return create_response(200, sign_types, cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_add_custom_sign_type(body, cors_headers):
    """Handle POST /api/asl-lex/sign-types"""
    try:
        result = asl_lex_service.add_custom_sign_type(body)
        return create_response(201, result, cors_headers)
    except Exception as e:
        return create_response(400, {'error': str(e)}, cors_headers)

def handle_get_custom_sign_types(cors_headers):
    """Handle GET /api/asl-lex/sign-types/custom"""
    try:
        custom_types = asl_lex_service.get_custom_sign_types()
        return create_response(200, custom_types, cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_batch_update_sign_types(body, cors_headers):
    """Handle POST /api/asl-lex/signs/batch-update-type"""
    try:
        result = asl_lex_service.batch_update_sign_types(body)
        return create_response(200, result, cors_headers)
    except Exception as e:
        return create_response(400, {'error': str(e)}, cors_headers)

def handle_validate_asl_sign(body, cors_headers):
    """Handle POST /api/asl-lex/validate-asl-sign"""
    try:
        validation_results = asl_lex_service.validate_asl_sign(body)
        return create_response(200, validation_results, cors_headers)
    except Exception as e:
        return create_response(400, {'error': str(e)}, cors_headers)

def handle_get_sign_type_analytics(cors_headers):
    """Handle GET /api/asl-lex/analytics/sign-types"""
    try:
        analytics = asl_lex_service.get_sign_type_analytics()
        return create_response(200, analytics, cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_bulk_upload(body, cors_headers):
    """Handle POST /api/asl-lex/bulk-upload"""
    try:
        result = asl_lex_service.create_bulk_upload_job(body)
        return create_response(202, result, cors_headers)
    except Exception as e:
        return create_response(400, {'error': str(e)}, cors_headers)

def handle_get_bulk_upload_jobs(cors_headers):
    """Handle GET /api/asl-lex/bulk-upload/jobs"""
    try:
        jobs = asl_lex_service.list_bulk_upload_jobs()
        return create_response(200, jobs, cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_get_bulk_upload_template(cors_headers):
    """Handle GET /api/asl-lex/bulk-upload/template"""
    try:
        template = asl_lex_service.get_bulk_upload_template()
        headers = cors_headers.copy()
        headers.update({
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename="asl_lex_bulk_upload_template.csv"'
        })
        return create_response(200, template, headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_get_statistics(cors_headers):
    """Handle GET /api/asl-lex/statistics"""
    try:
        stats = asl_lex_service.get_sign_statistics()
        return create_response(200, stats, cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_get_bulk_upload_job(job_id, cors_headers):
    """Handle GET /api/asl-lex/bulk-upload/jobs/{jobId}"""
    try:
        job = asl_lex_service.get_bulk_upload_job(job_id)
        if job:
            return create_response(200, job.__dict__, cors_headers)
        else:
            return create_response(404, {'error': 'Bulk upload job not found'}, cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_cancel_bulk_upload_job(job_id, cors_headers):
    """Handle POST /api/asl-lex/bulk-upload/jobs/{jobId}/cancel"""
    try:
        success = asl_lex_service.update_bulk_upload_job(job_id, {'status': 'cancelled'})
        if success:
            return create_response(200, {'message': 'Bulk upload job cancelled successfully'}, cors_headers)
        else:
            return create_response(404, {'error': 'Bulk upload job not found'}, cors_headers)
    except Exception as e:
        return create_response(500, {'error': str(e)}, cors_headers)

def handle_upload_video_with_metadata(body, cors_headers):
    """Handle POST /api/asl-lex/upload-video-with-metadata"""
    try:
        # This endpoint expects multipart form data with video file and metadata
        # The actual implementation would need to handle file uploads
        # For now, we'll create a placeholder that works with the frontend
        result = asl_lex_service.create_sign(body)
        return create_response(201, {
            'success': True,
            'sign_id': result.id,
            'message': 'Video uploaded successfully with metadata'
        }, cors_headers)
    except Exception as e:
        return create_response(400, {'error': str(e)}, cors_headers)

# Your existing handler functions (placeholders - replace with your actual implementations)
def handle_create_session(body, cors_headers):
    """Your existing session creation logic"""
    return create_response(200, {'message': 'Session created'}, cors_headers)

def handle_generate_upload_url(path_params, cors_headers):
    """Your existing upload URL generation logic"""
    return create_response(200, {'upload_url': 'https://example.com'}, cors_headers)

def handle_get_sessions(cors_headers):
    """Your existing sessions retrieval logic"""
    return create_response(200, {'sessions': []}, cors_headers)

def handle_get_annotations(path_params, cors_headers):
    """Your existing annotations retrieval logic"""
    return create_response(200, {'annotations': []}, cors_headers)

def handle_create_annotation(path_params, body, cors_headers):
    """Your existing annotation creation logic"""
    return create_response(200, {'message': 'Annotation created'}, cors_headers) 