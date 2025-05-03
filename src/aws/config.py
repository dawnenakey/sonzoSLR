import os
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

class AWSConfig:
    def __init__(self):
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = os.getenv('AWS_REGION', 'us-east-1')
        
        # Initialize AWS clients
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )
        
        # S3 bucket configuration
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'spokhand-data')
        
    def get_s3_client(self):
        return self.s3_client
    
    def get_bucket_name(self):
        return self.bucket_name 