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
    table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection'))
    
    # CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',  # Update this to your frontend domain in production
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Content-Type': 'application/json'
    }
    
    try:
        # Check if this is an API Gateway event
        if 'httpMethod' in event:
            print(f"HTTP Method: {event['httpMethod']}")
            print(f"Resource: {event.get('resource', 'No resource')}")
            print(f"Path: {event.get('path', 'No path')}")
            
            if event['httpMethod'] == 'OPTIONS':
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': ''
                }
            
            if event['httpMethod'] == 'POST' and event.get('resource') == '/sessions':
                # Parse the request body
                body = json.loads(event.get('body', '{}'))
                print(f"Request body: {body}")
                
                # Generate session ID and timestamp
                session_id = str(uuid.uuid4())
                timestamp = datetime.utcnow().isoformat()
                
                # Create session item
                session_item = {
                    'session_id': session_id,
                    'name': body.get('name', 'Untitled Session'),
                    'description': body.get('description', ''),
                    'created_at': timestamp,
                    'updated_at': timestamp,
                    'status': 'active'
                }
                
                # Save to DynamoDB
                table.put_item(Item=session_item)
                
                # Return success response with session details
                return json.dumps({
                    'success': True,
                    'session': {
                        'id': session_id,
                        'name': session_item['name'],
                        'description': session_item['description'],
                        'createdAt': session_item['created_at'],
                        'updatedAt': session_item['updated_at'],
                        'status': session_item['status']
                    }
                })
                
        # Return error for unsupported methods
        return json.dumps({
            'success': False,
            'error': 'Unsupported method or resource'
        })
                
    except Exception as e:
        # Return error response
        print(f"Error: {str(e)}")
        return json.dumps({
            'success': False,
            'error': str(e)
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