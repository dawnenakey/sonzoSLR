import json
import boto3
import os
import uuid
from datetime import datetime

def lambda_handler(event, context):
    """
    AWS Lambda function to handle session management and video processing
    """
    # Debug logging to see the event structure
    print(f"Event received: {json.dumps(event, indent=2)}")
    
    # Initialize DynamoDB client
    dynamodb = boto3.resource('dynamodb')
    table_name = os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection')
    table = dynamodb.Table(table_name)
    
    try:
        # Check if this is an API Gateway event
        if 'httpMethod' in event:
            print(f"HTTP Method: {event['httpMethod']}")
            print(f"Resource: {event.get('resource', 'No resource')}")
            print(f"Path: {event.get('path', 'No path')}")
            print(f"Path Parameters: {event.get('pathParameters', 'No path params')}")
            
            # Handle POST /sessions
            if event['httpMethod'] == 'POST' and event.get('path') == '/sessions':
                print("Creating new session...")
                
                # Parse the request body
                body = json.loads(event.get('body', '{}'))
                name = body.get('name', 'Unnamed Session')
                description = body.get('description', '')
                
                # Generate session ID and timestamp
                session_id = str(uuid.uuid4())
                timestamp = datetime.utcnow().isoformat()
                
                # Create session item for DynamoDB
                session_item = {
                    'sessionId': session_id,
                    'name': name,
                    'description': description,
                    'createdAt': timestamp,
                    'status': 'active',
                    'videoCount': 0
                }
                
                # Store in DynamoDB
                try:
                    table.put_item(Item=session_item)
                    print(f"Session created successfully: {session_id}")
                    
                    return json.dumps({
                        'success': True,
                        'session': {
                            'id': session_id,
                            'name': name,
                            'description': description,
                            'createdAt': timestamp,
                            'status': 'active'
                        }
                    })
                except Exception as db_error:
                    print(f"DynamoDB error: {db_error}")
                    return json.dumps({
                        'success': False,
                        'error': f'Database error: {str(db_error)}'
                    })
            
            # Handle GET /{sessionId}
            elif event['httpMethod'] == 'GET' and event.get('pathParameters') and 'sessionId' in event['pathParameters']:
                session_id = event['pathParameters']['sessionId']
                print(f"Getting session: {session_id}")
                
                try:
                    response = table.get_item(Key={'sessionId': session_id})
                    if 'Item' in response:
                        session = response['Item']
                        return json.dumps({
                            'success': True,
                            'session': session
                        })
                    else:
                        return json.dumps({
                            'success': False,
                            'error': 'Session not found'
                        })
                except Exception as db_error:
                    print(f"DynamoDB error: {db_error}")
                    return json.dumps({
                        'success': False,
                        'error': f'Database error: {str(db_error)}'
                    })
            
            # Handle OPTIONS requests for CORS
            elif event['httpMethod'] == 'OPTIONS':
                return json.dumps({
                    'success': True,
                    'message': 'CORS preflight handled'
                })
            
            else:
                print(f"Unsupported method/resource: {event['httpMethod']} {event.get('path')}")
                return json.dumps({
                    'success': False,
                    'error': f'Unsupported method or resource: {event["httpMethod"]} {event.get("path")}'
                })
        
        else:
            print("Not an API Gateway event")
            return json.dumps({
                'success': False,
                'error': 'Not an API Gateway event'
            })
    
    except Exception as e:
        print(f"General error: {e}")
        return json.dumps({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        })

    # Handle S3 events (for video processing)
    if 'Records' in event:
        # Initialize S3 client
        s3 = boto3.client('s3')
        bucket_name = os.environ['S3_BUCKET_NAME']
        
        # Get the uploaded file details from the event
        records = event.get('Records', [])
        for record in records:
            # Get the S3 object details
            s3_event = record.get('s3', {})
            bucket = s3_event.get('bucket', {}).get('name')
            key = s3_event.get('object', {}).get('key')
            
            if bucket == bucket_name:
                # Process the uploaded file
                response = s3.get_object(Bucket=bucket, Key=key)
                file_content = response['Body'].read()
                
                # TODO: Add your sign language processing logic here
                # For now, we'll just log the event
                print(f"Processing file: {key}")
                
                # Create a metadata file
                metadata = {
                    'filename': key,
                    'processed_at': datetime.now().isoformat(),
                    'status': 'processed'
                }
                
                # Save metadata back to S3
                metadata_key = f"metadata/{key}.json"
                s3.put_object(
                    Bucket=bucket_name,
                    Key=metadata_key,
                    Body=json.dumps(metadata)
                )
                
        return {
            'statusCode': 200,
            'body': json.dumps('Processing completed successfully')
        } 