import json
import logging

logger = logging.getLogger()

def lambda_handler(event, context):
    """
    Main Lambda handler for API Gateway requests.
    Routes requests to the appropriate handler based on HTTP method and path.
    """
    logger.info(f"--- Spokhand Lambda v{VERSION} ---")
    logger.info(f"Received event: {json.dumps(event)}")

    # Standard CORS headers
    cors_headers = {
        'Access-Control-Allow-Origin': 'https://main.djz1od5v7st0v.amplifyapp.com',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
    }

    try:
        http_method = event.get('httpMethod')
        resource = event.get('resource')  # Use resource for route matching
        path_params = event.get('pathParameters', {})

        if http_method == 'OPTIONS':
            return create_response(200, 'CORS preflight OK', cors_headers)

        elif http_method == 'POST' and resource == '/sessions':
            return handle_create_session(event, cors_headers)

        elif http_method == 'POST' and resource == '/sessions/{sessionId}/upload-video':
            return handle_generate_upload_url(event, cors_headers)

        elif http_method == 'GET' and resource == '/sessions':
            return handle_get_sessions(event, cors_headers)

        elif http_method == 'GET' and resource == '/videos/{videoId}/annotations':
            return handle_get_annotations(event, cors_headers)

        elif http_method == 'POST' and resource == '/videos/{videoId}/annotations':
            return handle_create_annotation(event, cors_headers)

        else:
            logger.warning(f"No route matched for method '{http_method}' and resource '{resource}'")
            return create_response(404, {'success': False, 'error': 'Not Found'}, cors_headers)

    except Exception as e:
        logger.error(f"!!! Unhandled exception: {str(e)}")
        return create_response(500, {'success': False, 'error': 'Internal Server Error'}, cors_headers) 