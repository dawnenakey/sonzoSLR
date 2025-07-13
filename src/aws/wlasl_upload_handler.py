import boto3
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class WLASLUploadHandler:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.s3_bucket = os.environ.get('S3_BUCKET', 'spokhand-data')
        self.table_name = os.environ.get('DYNAMODB_TABLE', 'spokhand-data-collection')
        self.table = self.dynamodb.Table(self.table_name)
    
    def upload_wlasl_video(self, video_file_path: str, metadata: Dict) -> Dict:
        """
        Upload a WLASL video file to S3 and store metadata in DynamoDB
        
        Args:
            video_file_path: Path to the .webm video file
            metadata: Dictionary containing video metadata (gloss, split, etc.)
            
        Returns:
            Dict containing upload results
        """
        try:
            # Generate unique ID for the video
            video_id = str(uuid.uuid4())
            
            # Create S3 key for the video
            s3_key = f"wlasl_videos/{metadata.get('split', 'unknown')}/{video_id}.webm"
            
            # Upload video to S3
            self.s3_client.upload_file(
                video_file_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={'ContentType': 'video/webm'}
            )
            
            # Generate presigned URL for streaming
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                ExpiresIn=3600  # 1 hour
            )
            
            # Prepare DynamoDB item
            timestamp = datetime.utcnow().isoformat()
            item = {
                'video_id': video_id,
                's3_key': s3_key,
                'bucket': self.s3_bucket,
                'presigned_url': presigned_url,
                'file_format': 'webm',
                'dataset': 'WLASL',
                'gloss': metadata.get('gloss', ''),
                'split': metadata.get('split', 'unknown'),  # train/val/test
                'instance_id': metadata.get('instance_id', ''),
                'video_url': metadata.get('video_url', ''),
                'bbox': metadata.get('bbox', []),
                'created_at': timestamp,
                'updated_at': timestamp,
                'status': 'uploaded'
            }
            
            # Store in DynamoDB
            self.table.put_item(Item=item)
            
            logger.info(f"Successfully uploaded WLASL video: {video_id}")
            
            return {
                'success': True,
                'video_id': video_id,
                's3_key': s3_key,
                'presigned_url': presigned_url,
                'metadata': item
            }
            
        except Exception as e:
            logger.error(f"Error uploading WLASL video: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_upload_wlasl_videos(self, video_metadata_list: List[Dict]) -> List[Dict]:
        """
        Upload multiple WLASL videos in batch
        
        Args:
            video_metadata_list: List of dictionaries containing file paths and metadata
            
        Returns:
            List of upload results
        """
        results = []
        
        for video_data in video_metadata_list:
            file_path = video_data.get('file_path')
            metadata = video_data.get('metadata', {})
            
            if file_path and os.path.exists(file_path):
                result = self.upload_wlasl_video(file_path, metadata)
                results.append(result)
            else:
                results.append({
                    'success': False,
                    'error': f"File not found: {file_path}"
                })
        
        return results
    
    def get_wlasl_videos(self, split: Optional[str] = None, gloss: Optional[str] = None) -> List[Dict]:
        """
        Retrieve WLASL videos from DynamoDB with optional filtering
        
        Args:
            split: Filter by split (train/val/test)
            gloss: Filter by gloss (sign name)
            
        Returns:
            List of video metadata
        """
        try:
            # Build scan parameters
            scan_kwargs = {
                'FilterExpression': 'dataset = :dataset',
                'ExpressionAttributeValues': {':dataset': 'WLASL'}
            }
            
            if split:
                scan_kwargs['FilterExpression'] += ' AND #split = :split'
                scan_kwargs['ExpressionAttributeNames'] = {'#split': 'split'}
                scan_kwargs['ExpressionAttributeValues'][':split'] = split
            
            if gloss:
                scan_kwargs['FilterExpression'] += ' AND gloss = :gloss'
                scan_kwargs['ExpressionAttributeValues'][':gloss'] = gloss
            
            response = self.table.scan(**scan_kwargs)
            return response.get('Items', [])
            
        except Exception as e:
            logger.error(f"Error retrieving WLASL videos: {str(e)}")
            return []
    
    def generate_streaming_url(self, video_id: str) -> Optional[str]:
        """
        Generate a new presigned URL for video streaming
        
        Args:
            video_id: The video ID
            
        Returns:
            Presigned URL or None if not found
        """
        try:
            response = self.table.get_item(Key={'video_id': video_id})
            item = response.get('Item')
            
            if item and 's3_key' in item:
                presigned_url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.s3_bucket, 'Key': item['s3_key']},
                    ExpiresIn=3600
                )
                
                # Update the presigned URL in DynamoDB
                self.table.update_item(
                    Key={'video_id': video_id},
                    UpdateExpression='SET presigned_url = :url, updated_at = :timestamp',
                    ExpressionAttributeValues={
                        ':url': presigned_url,
                        ':timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                return presigned_url
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating streaming URL: {str(e)}")
            return None 