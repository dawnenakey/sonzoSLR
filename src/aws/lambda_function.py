import json
import boto3
import os
from datetime import datetime

def lambda_handler(event, context):
    """
    AWS Lambda function to process sign language data
    """
    # Initialize S3 client
    s3 = boto3.client('s3')
    bucket_name = os.environ['S3_BUCKET_NAME']
    
    try:
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
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        } 