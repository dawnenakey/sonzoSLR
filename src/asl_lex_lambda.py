import json
import boto3
import os
from datetime import datetime
from asl_lex_service import ASLLexDataManager

# Initialize the ASL-LEX service
asl_lex_service = ASLLexDataManager()

def lambda_handler(event, context):
    """
    Lambda handler for ASL-LEX API Gateway integration
    """
    try:
        # Parse the API Gateway event
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '')
        path_parameters = event.get('pathParameters', {})
        query_string_parameters = event.get('queryStringParameters', {})
        body = event.get('body', '{}')
        
        # Parse body if it's a string
        if isinstance(body, str):
            try:
                body = json.loads(body) if body else {}
            except json.JSONDecodeError:
                body = {}
        
        # Route the request based on path and method
        response = route_request(http_method, path, path_parameters, query_string_parameters, body)
        
        return {
            'statusCode': response.get('status_code', 200),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
            },
            'body': json.dumps(response.get('data', {}))
        }
        
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def route_request(method, path, path_params, query_params, body):
    """
    Route the request to the appropriate handler based on path and method
    """
    
    # Handle CORS preflight requests
    if method == 'OPTIONS':
        return {'status_code': 200, 'data': {}}
    
    # Parse the path to determine the endpoint
    path_parts = path.strip('/').split('/')
    
    # API Gateway adds a stage name, so we need to handle that
    if len(path_parts) >= 3 and path_parts[0] == 'api' and path_parts[1] == 'asl-lex':
        endpoint = '/'.join(path_parts[2:])
    else:
        endpoint = '/'.join(path_parts)
    
    print(f"Routing request: {method} {endpoint}")
    
    # Route to appropriate handler
    if endpoint == 'signs' and method == 'GET':
        return handle_get_signs(query_params)
    elif endpoint == 'signs' and method == 'POST':
        return handle_create_sign(body)
    elif endpoint == 'sign-types' and method == 'GET':
        return handle_get_sign_types()
    elif endpoint == 'sign-types' and method == 'POST':
        return handle_add_custom_sign_type(body)
    elif endpoint == 'sign-types/custom' and method == 'GET':
        return handle_get_custom_sign_types()
    elif endpoint == 'signs/batch-update-type' and method == 'POST':
        return handle_batch_update_sign_types(body)
    elif endpoint == 'validate-asl-sign' and method == 'POST':
        return handle_validate_asl_sign(body)
    elif endpoint == 'analytics/sign-types' and method == 'GET':
        return handle_get_sign_type_analytics()
    elif endpoint == 'bulk-upload' and method == 'POST':
        return handle_bulk_upload(body)
    elif endpoint == 'bulk-upload/jobs' and method == 'GET':
        return handle_get_bulk_upload_jobs()
    elif endpoint == 'bulk-upload/template' and method == 'GET':
        return handle_get_bulk_upload_template()
    elif endpoint == 'statistics' and method == 'GET':
        return handle_get_statistics()
    else:
        return {
            'status_code': 404,
            'data': {'error': f'Endpoint not found: {method} {endpoint}'}
        }

def handle_get_signs(query_params):
    """Handle GET /api/asl-lex/signs"""
    try:
        # Convert query parameters to the format expected by the service
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
        return {'status_code': 200, 'data': [sign.__dict__ for sign in signs]}
    except Exception as e:
        return {'status_code': 500, 'data': {'error': str(e)}}

def handle_create_sign(body):
    """Handle POST /api/asl-lex/signs"""
    try:
        sign_id = asl_lex_service.create_sign(body)
        return {'status_code': 201, 'data': {'id': sign_id, 'message': 'Sign created successfully'}}
    except Exception as e:
        return {'status_code': 400, 'data': {'error': str(e)}}

def handle_get_sign_types():
    """Handle GET /api/asl-lex/sign-types"""
    try:
        sign_types = asl_lex_service.get_sign_types()
        return {'status_code': 200, 'data': sign_types}
    except Exception as e:
        return {'status_code': 500, 'data': {'error': str(e)}}

def handle_add_custom_sign_type(body):
    """Handle POST /api/asl-lex/sign-types"""
    try:
        result = asl_lex_service.add_custom_sign_type(body)
        return {'status_code': 201, 'data': result}
    except Exception as e:
        return {'status_code': 400, 'data': {'error': str(e)}}

def handle_get_custom_sign_types():
    """Handle GET /api/asl-lex/sign-types/custom"""
    try:
        custom_types = asl_lex_service.get_custom_sign_types()
        return {'status_code': 200, 'data': custom_types}
    except Exception as e:
        return {'status_code': 500, 'data': {'error': str(e)}}

def handle_batch_update_sign_types(body):
    """Handle POST /api/asl-lex/signs/batch-update-type"""
    try:
        result = asl_lex_service.batch_update_sign_types(body)
        return {'status_code': 200, 'data': result}
    except Exception as e:
        return {'status_code': 400, 'data': {'error': str(e)}}

def handle_validate_asl_sign(body):
    """Handle POST /api/asl-lex/validate-asl-sign"""
    try:
        validation_results = asl_lex_service.validate_asl_sign(body)
        return {'status_code': 200, 'data': validation_results}
    except Exception as e:
        return {'status_code': 400, 'data': {'error': str(e)}}

def handle_get_sign_type_analytics():
    """Handle GET /api/asl-lex/analytics/sign-types"""
    try:
        analytics = asl_lex_service.get_sign_type_analytics()
        return {'status_code': 200, 'data': analytics}
    except Exception as e:
        return {'status_code': 500, 'data': {'error': str(e)}}

def handle_bulk_upload(body):
    """Handle POST /api/asl-lex/bulk-upload"""
    try:
        # For Lambda, we'll need to handle file uploads differently
        # This is a placeholder - you might want to use S3 presigned URLs
        result = asl_lex_service.create_bulk_upload_job(body)
        return {'status_code': 202, 'data': result}
    except Exception as e:
        return {'status_code': 400, 'data': {'error': str(e)}}

def handle_get_bulk_upload_jobs():
    """Handle GET /api/asl-lex/bulk-upload/jobs"""
    try:
        jobs = asl_lex_service.list_bulk_upload_jobs()
        return {'status_code': 200, 'data': jobs}
    except Exception as e:
        return {'status_code': 500, 'data': {'error': str(e)}}

def handle_get_bulk_upload_template():
    """Handle GET /api/asl-lex/bulk-upload/template"""
    try:
        template = asl_lex_service.get_bulk_upload_template()
        return {
            'status_code': 200,
            'headers': {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename="asl_lex_bulk_upload_template.csv"'
            },
            'data': template
        }
    except Exception as e:
        return {'status_code': 500, 'data': {'error': str(e)}}

def handle_get_statistics():
    """Handle GET /api/asl-lex/statistics"""
    try:
        stats = asl_lex_service.get_statistics()
        return {'status_code': 200, 'data': stats}
    except Exception as e:
        return {'status_code': 500, 'data': {'error': str(e)}} 