import boto3
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ASLLexHandler:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.s3_bucket = os.environ.get('S3_BUCKET', 'spokhand-data')
        self.table_name = os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection')
        self.table = self.dynamodb.Table(self.table_name)
    
    def upload_asl_lex_csv(self, csv_file_path: str, metadata: Dict) -> Dict:
        """
        Upload an ASL-LEX CSV file to S3 and store metadata in DynamoDB
        
        Args:
            csv_file_path: Path to the CSV file
            metadata: Dictionary containing file metadata
            
        Returns:
            Dict containing upload results
        """
        try:
            # Generate unique ID for the file
            file_id = str(uuid.uuid4())
            
            # Create S3 key for the CSV file
            s3_key = f"asl_lex_data/{file_id}.csv"
            
            # Upload CSV to S3
            self.s3_client.upload_file(
                csv_file_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={'ContentType': 'text/csv'}
            )
            
            # Generate presigned URL for download
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                ExpiresIn=3600  # 1 hour
            )
            
            # Read CSV to get basic statistics
            df = pd.read_csv(csv_file_path)
            row_count = len(df)
            column_count = len(df.columns)
            columns = df.columns.tolist()
            
            # Prepare DynamoDB item
            timestamp = datetime.utcnow().isoformat()
            item = {
                'file_id': file_id,
                's3_key': s3_key,
                'bucket': self.s3_bucket,
                'presigned_url': presigned_url,
                'file_format': 'csv',
                'dataset': 'ASL-LEX',
                'filename': metadata.get('filename', os.path.basename(csv_file_path)),
                'description': metadata.get('description', ''),
                'row_count': row_count,
                'column_count': column_count,
                'columns': columns,
                'file_size': os.path.getsize(csv_file_path),
                'created_at': timestamp,
                'updated_at': timestamp,
                'status': 'uploaded'
            }
            
            # Store in DynamoDB
            self.table.put_item(Item=item)
            
            logger.info(f"Successfully uploaded ASL-LEX CSV: {file_id}")
            
            return {
                'success': True,
                'file_id': file_id,
                's3_key': s3_key,
                'presigned_url': presigned_url,
                'metadata': item,
                'statistics': {
                    'rows': row_count,
                    'columns': column_count,
                    'column_names': columns
                }
            }
            
        except Exception as e:
            logger.error(f"Error uploading ASL-LEX CSV: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_asl_lex_files(self) -> List[Dict]:
        """
        Retrieve all ASL-LEX CSV files from DynamoDB
        
        Returns:
            List of file metadata
        """
        try:
            response = self.table.scan(
                FilterExpression='dataset = :dataset',
                ExpressionAttributeValues={':dataset': 'ASL-LEX'}
            )
            return response.get('Items', [])
            
        except Exception as e:
            logger.error(f"Error retrieving ASL-LEX files: {str(e)}")
            return []
    
    def get_asl_lex_data(self, file_id: str) -> Optional[Dict]:
        """
        Retrieve ASL-LEX data from S3 and return as DataFrame
        
        Args:
            file_id: The file ID
            
        Returns:
            Dictionary containing data and metadata
        """
        try:
            # Get file metadata from DynamoDB
            response = self.table.get_item(Key={'file_id': file_id})
            item = response.get('Item')
            
            if not item:
                return None
            
            # Download CSV from S3
            csv_content = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=item['s3_key']
            )['Body'].read()
            
            # Parse CSV data
            df = pd.read_csv(pd.io.common.BytesIO(csv_content))
            
            return {
                'file_id': file_id,
                'metadata': item,
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'row_count': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving ASL-LEX data: {str(e)}")
            return None
    
    def generate_download_url(self, file_id: str) -> Optional[str]:
        """
        Generate a new presigned URL for CSV download
        
        Args:
            file_id: The file ID
            
        Returns:
            Presigned URL or None if not found
        """
        try:
            response = self.table.get_item(Key={'file_id': file_id})
            item = response.get('Item')
            
            if item and 's3_key' in item:
                presigned_url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.s3_bucket, 'Key': item['s3_key']},
                    ExpiresIn=3600
                )
                
                # Update the presigned URL in DynamoDB
                self.table.update_item(
                    Key={'file_id': file_id},
                    UpdateExpression='SET presigned_url = :url, updated_at = :timestamp',
                    ExpressionAttributeValues={
                        ':url': presigned_url,
                        ':timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                return presigned_url
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating download URL: {str(e)}")
            return None
    
    def search_asl_lex_data(self, file_id: str, search_term: str, column: Optional[str] = None) -> List[Dict]:
        """
        Search within ASL-LEX data
        
        Args:
            file_id: The file ID
            search_term: Term to search for
            column: Specific column to search in (optional)
            
        Returns:
            List of matching records
        """
        try:
            data_result = self.get_asl_lex_data(file_id)
            if not data_result:
                return []
            
            df = pd.DataFrame(data_result['data'])
            
            if column and column in df.columns:
                # Search in specific column
                mask = df[column].astype(str).str.contains(search_term, case=False, na=False)
            else:
                # Search across all columns
                mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            
            matching_records = df[mask].to_dict('records')
            
            return matching_records
            
        except Exception as e:
            logger.error(f"Error searching ASL-LEX data: {str(e)}")
            return [] 