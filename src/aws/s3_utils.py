import boto3
import os
from datetime import datetime
from config import AWSConfig

class S3Handler:
    def __init__(self):
        self.config = AWSConfig()
        self.s3_client = self.config.get_s3_client()
        self.bucket_name = self.config.get_bucket_name()
    
    def upload_file(self, file_path, s3_key=None):
        """
        Upload a file to S3
        
        Args:
            file_path (str): Path to the local file
            s3_key (str, optional): Key to use in S3. If None, will use the filename
            
        Returns:
            str: URL of the uploaded file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if s3_key is None:
            s3_key = os.path.basename(file_path)
        
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{timestamp}_{s3_key}"
        
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except Exception as e:
            raise Exception(f"Failed to upload file to S3: {str(e)}")
    
    def upload_data(self, data, s3_key):
        """
        Upload data directly to S3
        
        Args:
            data (bytes): Data to upload
            s3_key (str): Key to use in S3
            
        Returns:
            str: URL of the uploaded data
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data
            )
            return f"s3://{self.bucket_name}/{s3_key}"
        except Exception as e:
            raise Exception(f"Failed to upload data to S3: {str(e)}")
    
    def list_files(self, prefix=""):
        """
        List files in the S3 bucket
        
        Args:
            prefix (str): Prefix to filter files
            
        Returns:
            list: List of file keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            raise Exception(f"Failed to list files in S3: {str(e)}") 