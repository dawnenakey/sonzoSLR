#!/usr/bin/env python3
"""
Test script to verify presigned URL generation
"""

import json
import boto3
import os

def test_presigned_url_generation():
    """Test the presigned URL generation logic"""
    
    # Mock event similar to what API Gateway would send
    event = {
        'httpMethod': 'POST',
        'path': '/sessions/test-session-id/upload-video',
        'pathParameters': {
            'sessionId': 'test-session-id'
        },
        'headers': {
            'content-type': 'application/json'
        },
        'body': json.dumps({
            'filename': 'test_video.mp4',
            'contentType': 'video/mp4'
        })
    }
    
    # Initialize S3 client
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET', 'spokhand-data')
    s3_client = boto3.client('s3')
    
    try:
        # Extract parameters from event
        session_id = event.get('pathParameters', {}).get('sessionId')
        body = event.get('body', '{}')
        content_type = event.get('headers', {}).get('content-type', '')
        
        print(f"Session ID: {session_id}")
        print(f"Content-Type: {content_type}")
        print(f"Body: {body}")
        
        filename = None
        file_content_type = None
        
        if 'application/json' in content_type:
            try:
                body_data = json.loads(body) if body else {}
                filename = body_data.get('filename')
                file_content_type = body_data.get('contentType')
                print(f"Parsed filename: {filename}")
                print(f"Parsed content type: {file_content_type}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return
        
        if not filename:
            print("Error: filename is required")
            return
        
        if not file_content_type:
            file_content_type = 'video/mp4'
        
        # Generate object key
        object_key = f"uploads/{session_id}/{filename}"
        print(f"Object key: {object_key}")
        
        # Generate presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': object_key, 'ContentType': file_content_type},
            ExpiresIn=3600
        )
        
        print(f"Generated presigned URL: {presigned_url}")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_presigned_url_generation() 