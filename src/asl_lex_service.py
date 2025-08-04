"""
ASL-LEX Data Management Service

This service handles ASL-LEX sign data management including:
- Uploading sign videos to AWS S3
- Storing sign metadata in DynamoDB
- Managing sign validation and approval workflows
- Providing search and filtering capabilities
- Bulk upload of CSV spreadsheets and ZIP files
"""

import boto3
import json
import uuid
import logging
import zipfile
import csv
import io
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from botocore.exceptions import ClientError
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ASLLexSign:
    """Represents an ASL-LEX sign with all metadata."""
    id: str
    gloss: str
    english: str
    handshape: str
    location: str
    movement: str
    palm_orientation: str
    dominant_hand: str
    non_dominant_hand: str
    video_url: str
    sign_type: str  # isolated_sign, continuous_signing, fingerspelling, classifier, compound_sign, etc.
    frequency: int
    age_of_acquisition: float
    iconicity: float
    lexical_class: str
    tags: List[str]
    notes: str
    uploaded_by: str
    uploaded_at: str
    status: str  # pending, approved, rejected
    confidence_score: float
    validation_status: str  # unvalidated, validated, needs_review

@dataclass
class BulkUploadJob:
    """Represents a bulk upload job."""
    id: str
    filename: str
    file_type: str  # csv, zip
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    status: str  # processing, completed, failed
    uploaded_by: str
    uploaded_at: str
    error_log: List[str]
    metadata: Dict[str, Any]

class ASLLexDataManager:
    """Manages ASL-LEX sign data in AWS S3 and DynamoDB."""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.s3_bucket = os.getenv('S3_BUCKET', 'spokhand-data')
        self.table_name = os.getenv('DYNAMODB_TABLE', 'spokhand-data-collection')
        self.table = self.dynamodb.Table(self.table_name)
        
    def upload_sign_video(self, video_data: bytes, filename: str, sign_id: str) -> str:
        """Upload sign video to S3 and return the URL."""
        try:
            # Create S3 key for the video
            s3_key = f"asl-lex/signs/{sign_id}/{filename}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=video_data,
                ContentType='video/mp4',
                Metadata={
                    'sign_id': sign_id,
                    'uploaded_at': datetime.utcnow().isoformat(),
                    'content_type': 'asl_lex_sign'
                }
            )
            
            # Generate public URL
            video_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            logger.info(f"Uploaded sign video: {video_url}")
            return video_url
            
        except Exception as e:
            logger.error(f"Error uploading sign video: {str(e)}")
            raise Exception(f"Failed to upload sign video: {str(e)}")
    
    def upload_bulk_file(self, file_data: bytes, filename: str, file_type: str, uploaded_by: str) -> str:
        """Upload a bulk file (CSV or ZIP) to S3 and return the URL."""
        try:
            # Create S3 key for the bulk file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            s3_key = f"asl-lex/bulk-uploads/{timestamp}_{filename}"
            
            # Determine content type
            content_type = 'text/csv' if file_type == 'csv' else 'application/zip'
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=file_data,
                ContentType=content_type,
                Metadata={
                    'file_type': file_type,
                    'uploaded_by': uploaded_by,
                    'uploaded_at': datetime.utcnow().isoformat(),
                    'content_type': 'bulk_upload'
                }
            )
            
            # Generate public URL
            file_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            logger.info(f"Uploaded bulk file: {file_url}")
            return file_url
            
        except Exception as e:
            logger.error(f"Error uploading bulk file: {str(e)}")
            raise Exception(f"Failed to upload bulk file: {str(e)}")
    
    def create_bulk_upload_job(self, filename: str, file_type: str, uploaded_by: str, total_items: int = 0) -> BulkUploadJob:
        """Create a new bulk upload job."""
        try:
            job_id = f"bulk-{uuid.uuid4().hex[:8]}"
            
            job = BulkUploadJob(
                id=job_id,
                filename=filename,
                file_type=file_type,
                total_items=total_items,
                processed_items=0,
                successful_items=0,
                failed_items=0,
                status='processing',
                uploaded_by=uploaded_by,
                uploaded_at=datetime.utcnow().isoformat(),
                error_log=[],
                metadata={}
            )
            
            # Store in DynamoDB
            item = asdict(job)
            item['timestamp'] = job.uploaded_at  # Use as sort key
            item['content_type'] = 'bulk_upload_job'
            
            self.table.put_item(Item=item)
            logger.info(f"Created bulk upload job: {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error creating bulk upload job: {str(e)}")
            raise Exception(f"Failed to create bulk upload job: {str(e)}")
    
    def update_bulk_upload_job(self, job_id: str, updates: Dict) -> Optional[BulkUploadJob]:
        """Update a bulk upload job."""
        try:
            # Get current job
            job = self.get_bulk_upload_job(job_id)
            if not job:
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            # Update in DynamoDB
            update_expression = "SET "
            expression_values = {}
            
            for key, value in updates.items():
                if hasattr(job, key):
                    update_expression += f"{key} = :{key}, "
                    expression_values[f":{key}"] = value
            
            update_expression = update_expression.rstrip(", ")
            
            self.table.update_item(
                Key={'id': job_id, 'timestamp': job.uploaded_at},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            
            logger.info(f"Updated bulk upload job: {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Error updating bulk upload job {job_id}: {str(e)}")
            return None
    
    def get_bulk_upload_job(self, job_id: str) -> Optional[BulkUploadJob]:
        """Get a bulk upload job by ID."""
        try:
            response = self.table.query(
                KeyConditionExpression='id = :id',
                FilterExpression='content_type = :content_type',
                ExpressionAttributeValues={
                    ':id': job_id,
                    ':content_type': 'bulk_upload_job'
                }
            )
            
            if response['Items']:
                item = response['Items'][0]
                return BulkUploadJob(**item)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving bulk upload job {job_id}: {str(e)}")
            return None
    
    def list_bulk_upload_jobs(self, status_filter: str = 'all') -> List[BulkUploadJob]:
        """List all bulk upload jobs with optional filtering."""
        try:
            # Scan the table for bulk upload jobs
            response = self.table.scan(
                FilterExpression='content_type = :content_type',
                ExpressionAttributeValues={
                    ':content_type': 'bulk_upload_job'
                }
            )
            
            jobs = []
            for item in response['Items']:
                job = BulkUploadJob(**item)
                
                # Apply filters
                if status_filter != 'all' and job.status != status_filter:
                    continue
                
                jobs.append(job)
            
            # Sort by upload date (newest first)
            jobs.sort(key=lambda x: x.uploaded_at, reverse=True)
            return jobs
            
        except Exception as e:
            logger.error(f"Error listing bulk upload jobs: {str(e)}")
            return []
    
    def process_csv_bulk_upload(self, csv_data: str, job_id: str, uploaded_by: str) -> Dict[str, Any]:
        """Process a CSV bulk upload."""
        try:
            job = self.get_bulk_upload_job(job_id)
            if not job:
                raise Exception("Bulk upload job not found")
            
            # Parse CSV data
            reader = csv.DictReader(io.StringIO(csv_data))
            rows = list(reader)
            
            # Update job with total items
            self.update_bulk_upload_job(job_id, {'total_items': len(rows)})
            
            successful_items = []
            failed_items = []
            error_log = []
            
            for i, row in enumerate(rows):
                try:
                    # Convert string values to appropriate types
                    if 'frequency' in row:
                        row['frequency'] = int(row['frequency']) if row['frequency'] else 0
                    if 'age_of_acquisition' in row:
                        row['age_of_acquisition'] = float(row['age_of_acquisition']) if row['age_of_acquisition'] else 0.0
                    if 'iconicity' in row:
                        row['iconicity'] = float(row['iconicity']) if row['iconicity'] else 0.0
                    if 'confidence_score' in row:
                        row['confidence_score'] = float(row['confidence_score']) if row['confidence_score'] else 0.0
                    if 'tags' in row and row['tags']:
                        row['tags'] = [tag.strip() for tag in row['tags'].split(',')]
                    
                    # Add uploaded_by if not present
                    if 'uploaded_by' not in row:
                        row['uploaded_by'] = uploaded_by
                    
                    # Create sign
                    sign = self.create_sign(row)
                    successful_items.append(sign.id)
                    
                    # Update job progress
                    self.update_bulk_upload_job(job_id, {
                        'processed_items': i + 1,
                        'successful_items': len(successful_items)
                    })
                    
                except Exception as e:
                    error_msg = f"Row {i + 1}: {str(e)}"
                    error_log.append(error_msg)
                    failed_items.append(i + 1)
                    
                    # Update job progress
                    self.update_bulk_upload_job(job_id, {
                        'processed_items': i + 1,
                        'failed_items': len(failed_items),
                        'error_log': error_log
                    })
            
            # Mark job as completed
            final_status = 'completed' if len(failed_items) == 0 else 'completed_with_errors'
            self.update_bulk_upload_job(job_id, {
                'status': final_status,
                'processed_items': len(rows),
                'successful_items': len(successful_items),
                'failed_items': len(failed_items),
                'error_log': error_log
            })
            
            return {
                'job_id': job_id,
                'total_items': len(rows),
                'successful_items': len(successful_items),
                'failed_items': len(failed_items),
                'error_log': error_log,
                'status': final_status
            }
            
        except Exception as e:
            logger.error(f"Error processing CSV bulk upload: {str(e)}")
            self.update_bulk_upload_job(job_id, {
                'status': 'failed',
                'error_log': [str(e)]
            })
            raise Exception(f"Failed to process CSV bulk upload: {str(e)}")
    
    def process_zip_bulk_upload(self, zip_data: bytes, job_id: str, uploaded_by: str) -> Dict[str, Any]:
        """Process a ZIP bulk upload containing videos and metadata."""
        try:
            job = self.get_bulk_upload_job(job_id)
            if not job:
                raise Exception("Bulk upload job not found")
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP file
                with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
                    zip_file.extractall(temp_dir)
                
                # Look for CSV file with metadata
                csv_file = None
                video_files = []
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.lower().endswith('.csv'):
                            csv_file = file_path
                        elif file.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                            video_files.append(file_path)
                
                if not csv_file:
                    raise Exception("No CSV file found in ZIP")
                
                # Read CSV metadata
                with open(csv_file, 'r', encoding='utf-8') as f:
                    csv_data = f.read()
                
                # Parse CSV to get metadata
                reader = csv.DictReader(io.StringIO(csv_data))
                metadata_rows = list(reader)
                
                # Update job with total items
                self.update_bulk_upload_job(job_id, {'total_items': len(metadata_rows)})
                
                successful_items = []
                failed_items = []
                error_log = []
                
                for i, row in enumerate(metadata_rows):
                    try:
                        # Look for corresponding video file
                        video_filename = row.get('video_filename', '')
                        video_path = None
                        
                        if video_filename:
                            for video_file in video_files:
                                if os.path.basename(video_file) == video_filename:
                                    video_path = video_file
                                    break
                        
                        # Convert string values to appropriate types
                        if 'frequency' in row:
                            row['frequency'] = int(row['frequency']) if row['frequency'] else 0
                        if 'age_of_acquisition' in row:
                            row['age_of_acquisition'] = float(row['age_of_acquisition']) if row['age_of_acquisition'] else 0.0
                        if 'iconicity' in row:
                            row['iconicity'] = float(row['iconicity']) if row['iconicity'] else 0.0
                        if 'confidence_score' in row:
                            row['confidence_score'] = float(row['confidence_score']) if row['confidence_score'] else 0.0
                        if 'tags' in row and row['tags']:
                            row['tags'] = [tag.strip() for tag in row['tags'].split(',')]
                        
                        # Add uploaded_by if not present
                        if 'uploaded_by' not in row:
                            row['uploaded_by'] = uploaded_by
                        
                        # Create sign first
                        sign = self.create_sign(row)
                        
                        # Upload video if found
                        if video_path and os.path.exists(video_path):
                            with open(video_path, 'rb') as f:
                                video_data = f.read()
                            
                            video_url = self.upload_sign_video(video_data, video_filename, sign.id)
                            self.update_sign(sign.id, {'video_url': video_url})
                        
                        successful_items.append(sign.id)
                        
                        # Update job progress
                        self.update_bulk_upload_job(job_id, {
                            'processed_items': i + 1,
                            'successful_items': len(successful_items)
                        })
                        
                    except Exception as e:
                        error_msg = f"Row {i + 1}: {str(e)}"
                        error_log.append(error_msg)
                        failed_items.append(i + 1)
                        
                        # Update job progress
                        self.update_bulk_upload_job(job_id, {
                            'processed_items': i + 1,
                            'failed_items': len(failed_items),
                            'error_log': error_log
                        })
                
                # Mark job as completed
                final_status = 'completed' if len(failed_items) == 0 else 'completed_with_errors'
                self.update_bulk_upload_job(job_id, {
                    'status': final_status,
                    'processed_items': len(metadata_rows),
                    'successful_items': len(successful_items),
                    'failed_items': len(failed_items),
                    'error_log': error_log
                })
                
                return {
                    'job_id': job_id,
                    'total_items': len(metadata_rows),
                    'successful_items': len(successful_items),
                    'failed_items': len(failed_items),
                    'error_log': error_log,
                    'status': final_status
                }
                
        except Exception as e:
            logger.error(f"Error processing ZIP bulk upload: {str(e)}")
            self.update_bulk_upload_job(job_id, {
                'status': 'failed',
                'error_log': [str(e)]
            })
            raise Exception(f"Failed to process ZIP bulk upload: {str(e)}")
    
    def create_sign(self, sign_data: Dict) -> ASLLexSign:
        """Create a new ASL-LEX sign in the database."""
        try:
            sign_id = f"asl-{uuid.uuid4().hex[:8]}"
            
            # Create sign object
            sign = ASLLexSign(
                id=sign_id,
                gloss=sign_data['gloss'],
                english=sign_data['english'],
                handshape=sign_data.get('handshape', ''),
                location=sign_data.get('location', ''),
                movement=sign_data.get('movement', ''),
                palm_orientation=sign_data.get('palm_orientation', ''),
                dominant_hand=sign_data.get('dominant_hand', ''),
                non_dominant_hand=sign_data.get('non_dominant_hand', ''),
                video_url=sign_data.get('video_url', ''),
                sign_type=sign_data.get('sign_type', 'isolated_sign'), # Default to isolated_sign
                frequency=sign_data.get('frequency', 0),
                age_of_acquisition=sign_data.get('age_of_acquisition', 0.0),
                iconicity=sign_data.get('iconicity', 0.0),
                lexical_class=sign_data.get('lexical_class', ''),
                tags=sign_data.get('tags', []),
                notes=sign_data.get('notes', ''),
                uploaded_by=sign_data.get('uploaded_by', 'unknown'),
                uploaded_at=datetime.utcnow().isoformat(),
                status='pending',
                confidence_score=sign_data.get('confidence_score', 0.0),
                validation_status='unvalidated'
            )
            
            # Store in DynamoDB
            item = asdict(sign)
            item['timestamp'] = sign.uploaded_at  # Use as sort key
            item['content_type'] = 'asl_lex_sign'
            
            self.table.put_item(Item=item)
            logger.info(f"Created ASL-LEX sign: {sign_id}")
            return sign
            
        except Exception as e:
            logger.error(f"Error creating ASL-LEX sign: {str(e)}")
            raise Exception(f"Failed to create sign: {str(e)}")
    
    def get_sign(self, sign_id: str) -> Optional[ASLLexSign]:
        """Retrieve a sign by ID."""
        try:
            response = self.table.query(
                KeyConditionExpression='id = :id',
                FilterExpression='content_type = :content_type',
                ExpressionAttributeValues={
                    ':id': sign_id,
                    ':content_type': 'asl_lex_sign'
                }
            )
            
            if response['Items']:
                item = response['Items'][0]
                return ASLLexSign(**item)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving sign {sign_id}: {str(e)}")
            return None
    
    def list_signs(self, 
                   status_filter: str = 'all',
                   handshape_filter: str = 'all',
                   location_filter: str = 'all',
                   search_term: str = '') -> List[ASLLexSign]:
        """List all signs with optional filtering."""
        try:
            # Scan the table for ASL-LEX signs
            response = self.table.scan(
                FilterExpression='content_type = :content_type',
                ExpressionAttributeValues={
                    ':content_type': 'asl_lex_sign'
                }
            )
            
            signs = []
            for item in response['Items']:
                sign = ASLLexSign(**item)
                
                # Apply filters
                if status_filter != 'all' and sign.status != status_filter:
                    continue
                if handshape_filter != 'all' and sign.handshape != handshape_filter:
                    continue
                if location_filter != 'all' and sign.location != location_filter:
                    continue
                if search_term and search_term.lower() not in sign.gloss.lower() and search_term.lower() not in sign.english.lower():
                    continue
                
                signs.append(sign)
            
            # Sort by upload date (newest first)
            signs.sort(key=lambda x: x.uploaded_at, reverse=True)
            return signs
            
        except Exception as e:
            logger.error(f"Error listing signs: {str(e)}")
            return []
    
    def update_sign(self, sign_id: str, updates: Dict) -> Optional[ASLLexSign]:
        """Update a sign's metadata."""
        try:
            # Get current sign
            sign = self.get_sign(sign_id)
            if not sign:
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(sign, key):
                    setattr(sign, key, value)
            
            # Update in DynamoDB
            update_expression = "SET "
            expression_values = {}
            
            for key, value in updates.items():
                if hasattr(sign, key):
                    update_expression += f"{key} = :{key}, "
                    expression_values[f":{key}"] = value
            
            update_expression = update_expression.rstrip(", ")
            
            self.table.update_item(
                Key={'id': sign_id, 'timestamp': sign.uploaded_at},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            
            logger.info(f"Updated ASL-LEX sign: {sign_id}")
            return sign
            
        except Exception as e:
            logger.error(f"Error updating sign {sign_id}: {str(e)}")
            return None
    
    def delete_sign(self, sign_id: str) -> bool:
        """Delete a sign from the database."""
        try:
            sign = self.get_sign(sign_id)
            if not sign:
                return False
            
            # Delete from DynamoDB
            self.table.delete_item(
                Key={'id': sign_id, 'timestamp': sign.uploaded_at}
            )
            
            # Optionally delete video from S3
            if sign.video_url and 's3.amazonaws.com' in sign.video_url:
                try:
                    # Extract S3 key from URL
                    url_parts = sign.video_url.split('/')
                    s3_key = '/'.join(url_parts[3:])  # Skip https://bucket.s3.amazonaws.com/
                    self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                    logger.info(f"Deleted video from S3: {s3_key}")
                except Exception as e:
                    logger.warning(f"Could not delete video from S3: {str(e)}")
            
            logger.info(f"Deleted ASL-LEX sign: {sign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting sign {sign_id}: {str(e)}")
            return False
    
    def get_sign_statistics(self) -> Dict:
        """Get statistics about the ASL-LEX database."""
        try:
            signs = self.list_signs()
            
            stats = {
                'total': len(signs),
                'approved': len([s for s in signs if s.status == 'approved']),
                'pending': len([s for s in signs if s.status == 'pending']),
                'rejected': len([s for s in signs if s.status == 'rejected']),
                'validated': len([s for s in signs if s.validation_status == 'validated']),
                'unvalidated': len([s for s in signs if s.validation_status == 'unvalidated']),
                'needs_review': len([s for s in signs if s.validation_status == 'needs_review']),
                'avg_confidence': sum(s.confidence_score for s in signs) / len(signs) if signs else 0,
                'handshapes': {},
                'locations': {},
                'lexical_classes': {}
            }
            
            # Count handshapes, locations, and lexical classes
            for sign in signs:
                if sign.handshape:
                    stats['handshapes'][sign.handshape] = stats['handshapes'].get(sign.handshape, 0) + 1
                if sign.location:
                    stats['locations'][sign.location] = stats['locations'].get(sign.location, 0) + 1
                if sign.lexical_class:
                    stats['lexical_classes'][sign.lexical_class] = stats['lexical_classes'].get(sign.lexical_class, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}

    def get_sign_types(self) -> List[Dict]:
        """Get list of valid sign types for classification."""
        try:
            sign_types = [
                {
                    'value': 'isolated_sign',
                    'label': 'Isolated Sign',
                    'description': 'Single sign performed in isolation'
                },
                {
                    'value': 'continuous_signing',
                    'label': 'Continuous Signing',
                    'description': 'Sign within continuous signing context'
                },
                {
                    'value': 'fingerspelling',
                    'label': 'Fingerspelling',
                    'description': 'Manual alphabet or fingerspelling'
                },
                {
                    'value': 'classifier',
                    'label': 'Classifier',
                    'description': 'Classifier construction or classifier predicate'
                },
                {
                    'value': 'compound_sign',
                    'label': 'Compound Sign',
                    'description': 'Compound sign combining multiple elements'
                },
                {
                    'value': 'inflected_sign',
                    'label': 'Inflected Sign',
                    'description': 'Sign with grammatical inflection'
                },
                {
                    'value': 'directional_sign',
                    'label': 'Directional Sign',
                    'description': 'Sign with directional movement'
                },
                {
                    'value': 'spatial_sign',
                    'label': 'Spatial Sign',
                    'description': 'Sign involving spatial relationships'
                },
                {
                    'value': 'temporal_sign',
                    'label': 'Temporal Sign',
                    'description': 'Sign indicating time or temporal aspect'
                },
                {
                    'value': 'manner_sign',
                    'label': 'Manner Sign',
                    'description': 'Sign indicating manner of action'
                },
                {
                    'value': 'number_sign',
                    'label': 'Number Sign',
                    'description': 'Number or numeral sign'
                },
                {
                    'value': 'question_sign',
                    'label': 'Question Sign',
                    'description': 'Question or interrogative sign'
                },
                {
                    'value': 'negation_sign',
                    'label': 'Negation Sign',
                    'description': 'Negative or negation sign'
                },
                {
                    'value': 'modality_sign',
                    'label': 'Modality Sign',
                    'description': 'Sign indicating modality or mood'
                },
                {
                    'value': 'other',
                    'label': 'Other',
                    'description': 'Other type of sign not listed above'
                }
            ]
            
            return sign_types
            
        except Exception as e:
            logger.error(f"Error getting sign types: {str(e)}")
            return []

    def get_custom_sign_types(self) -> List[Dict]:
        """Get list of custom sign types."""
        try:
            response = self.table.scan(
                FilterExpression='content_type = :content_type',
                ExpressionAttributeValues={
                    ':content_type': 'custom_sign_type'
                }
            )
            
            custom_types = []
            for item in response['Items']:
                custom_types.append({
                    'value': item['value'],
                    'label': item['label'],
                    'description': item.get('description', ''),
                    'is_custom': True,
                    'created_by': item.get('created_by', 'unknown'),
                    'created_at': item.get('created_at', '')
                })
            
            return custom_types
            
        except Exception as e:
            logger.error(f"Error getting custom sign types: {str(e)}")
            return []

    def get_bulk_upload_template(self) -> str:
        """Get a CSV template for bulk upload."""
        try:
            # Create CSV template with all required fields
            template_data = [
                {
                    'gloss': 'HELLO',
                    'english': 'Hello',
                    'handshape': 'B',
                    'location': 'neutral space',
                    'movement': 'wave',
                    'palm_orientation': 'palm forward',
                    'dominant_hand': 'B',
                    'non_dominant_hand': 'B',
                    'video_filename': 'hello.mp4',
                    'sign_type': 'isolated_sign',
                    'frequency': 100,
                    'age_of_acquisition': 2.5,
                    'iconicity': 0.8,
                    'lexical_class': 'interjection',
                    'tags': 'greeting,common',
                    'notes': 'Common greeting sign',
                    'uploaded_by': 'data_analyst'
                },
                {
                    'gloss': 'THANK-YOU',
                    'english': 'Thank you',
                    'handshape': 'A',
                    'location': 'chin',
                    'movement': 'forward',
                    'palm_orientation': 'palm in',
                    'dominant_hand': 'A',
                    'non_dominant_hand': 'B',
                    'video_filename': 'thank_you.mp4',
                    'sign_type': 'isolated_sign',
                    'frequency': 85,
                    'age_of_acquisition': 2.0,
                    'iconicity': 0.6,
                    'lexical_class': 'interjection',
                    'tags': 'politeness,common',
                    'notes': 'Polite expression of gratitude',
                    'uploaded_by': 'data_analyst'
                }
            ]
            
            # Convert to CSV
            output = io.StringIO()
            if template_data:
                writer = csv.DictWriter(output, fieldnames=template_data[0].keys())
                writer.writeheader()
                writer.writerows(template_data)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error getting bulk upload template: {str(e)}")
            return ""

    def get_sign_type_analytics(self) -> Dict:
        """Get analytics on sign type distribution."""
        try:
            signs = self.list_signs()
            
            # Count by sign type
            sign_type_counts = {}
            sign_type_details = {}
            
            for sign in signs:
                sign_type = sign.sign_type or 'unknown'
                if sign_type not in sign_type_counts:
                    sign_type_counts[sign_type] = 0
                    sign_type_details[sign_type] = {
                        'count': 0,
                        'examples': [],
                        'avg_confidence': 0,
                        'validation_status': {'validated': 0, 'unvalidated': 0, 'needs_review': 0}
                    }
                
                sign_type_counts[sign_type] += 1
                sign_type_details[sign_type]['count'] += 1
                
                # Add example (limit to 5)
                if len(sign_type_details[sign_type]['examples']) < 5:
                    sign_type_details[sign_type]['examples'].append({
                        'gloss': sign.gloss,
                        'english': sign.english,
                        'id': sign.id
                    })
                
                # Track validation status
                validation_status = sign.validation_status or 'unvalidated'
                sign_type_details[sign_type]['validation_status'][validation_status] += 1
            
            # Calculate averages and percentages
            total_signs = len(signs)
            sign_type_percentages = {}
            
            for sign_type, count in sign_type_counts.items():
                percentage = (count / total_signs * 100) if total_signs > 0 else 0
                sign_type_percentages[sign_type] = round(percentage, 2)
                
                # Calculate average confidence for this sign type
                type_signs = [s for s in signs if s.sign_type == sign_type]
                if type_signs:
                    avg_confidence = sum(s.confidence_score for s in type_signs) / len(type_signs)
                    sign_type_details[sign_type]['avg_confidence'] = round(avg_confidence, 3)
            
            # Get top sign types
            top_sign_types = sorted(sign_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Get recent trends (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_signs = [s for s in signs if datetime.fromisoformat(s.uploaded_at) > thirty_days_ago]
            
            recent_sign_type_counts = {}
            for sign in recent_signs:
                sign_type = sign.sign_type or 'unknown'
                recent_sign_type_counts[sign_type] = recent_sign_type_counts.get(sign_type, 0) + 1
            
            return {
                'total_signs': total_signs,
                'sign_type_counts': sign_type_counts,
                'sign_type_percentages': sign_type_percentages,
                'sign_type_details': sign_type_details,
                'top_sign_types': top_sign_types,
                'recent_trends': recent_sign_type_counts,
                'custom_types': self.get_custom_sign_types()
            }
            
        except Exception as e:
            logger.error(f"Error getting sign type analytics: {str(e)}")
            return {}
    
    def validate_sign(self, sign_id: str, validator: str, is_valid: bool, notes: str = '') -> bool:
        """Validate a sign by a data analyst."""
        try:
            updates = {
                'validation_status': 'validated' if is_valid else 'needs_review',
                'validated_by': validator,
                'validated_at': datetime.utcnow().isoformat(),
                'validation_notes': notes
            }
            
            if is_valid:
                updates['status'] = 'approved'
            
            return self.update_sign(sign_id, updates) is not None
            
        except Exception as e:
            logger.error(f"Error validating sign {sign_id}: {str(e)}")
            return False
    
    def export_signs(self, format: str = 'json') -> str:
        """Export all signs in the specified format."""
        try:
            signs = self.list_signs()
            
            if format == 'json':
                return json.dumps([asdict(sign) for sign in signs], indent=2)
            elif format == 'csv':
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                if signs:
                    writer.writerow(asdict(signs[0]).keys())
                
                # Write data
                for sign in signs:
                    writer.writerow(asdict(sign).values())
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting signs: {str(e)}")
            return ""
    
    def import_signs(self, data: str, format: str = 'json') -> List[str]:
        """Import signs from external data."""
        try:
            imported_ids = []
            
            if format == 'json':
                signs_data = json.loads(data)
                for sign_data in signs_data:
                    sign = self.create_sign(sign_data)
                    imported_ids.append(sign.id)
            elif format == 'csv':
                import csv
                import io
                
                reader = csv.DictReader(io.StringIO(data))
                for row in reader:
                    # Convert string values to appropriate types
                    if 'frequency' in row:
                        row['frequency'] = int(row['frequency']) if row['frequency'] else 0
                    if 'age_of_acquisition' in row:
                        row['age_of_acquisition'] = float(row['age_of_acquisition']) if row['age_of_acquisition'] else 0.0
                    if 'iconicity' in row:
                        row['iconicity'] = float(row['iconicity']) if row['iconicity'] else 0.0
                    if 'confidence_score' in row:
                        row['confidence_score'] = float(row['confidence_score']) if row['confidence_score'] else 0.0
                    if 'tags' in row and row['tags']:
                        row['tags'] = [tag.strip() for tag in row['tags'].split(',')]
                    
                    sign = self.create_sign(row)
                    imported_ids.append(sign.id)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            logger.info(f"Imported {len(imported_ids)} signs")
            return imported_ids
            
        except Exception as e:
            logger.error(f"Error importing signs: {str(e)}")
            return []

# Flask API endpoints for ASL-LEX management
def create_asl_lex_routes(app):
    """Create Flask routes for ASL-LEX data management."""
    
    # Import Flask components
    from flask import request, jsonify, Response
    
    manager = ASLLexDataManager()
    
    @app.route('/api/asl-lex/signs', methods=['GET'])
    def list_asl_lex_signs():
        """List all ASL-LEX signs with optional filtering."""
        try:
            status_filter = request.args.get('status', 'all')
            handshape_filter = request.args.get('handshape', 'all')
            location_filter = request.args.get('location', 'all')
            search_term = request.args.get('search', '')
            
            signs = manager.list_signs(
                status_filter=status_filter,
                handshape_filter=handshape_filter,
                location_filter=location_filter,
                search_term=search_term
            )
            
            return jsonify([asdict(sign) for sign in signs])
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/bulk-upload', methods=['POST'])
    def bulk_upload():
        """Upload and process CSV or ZIP files in bulk."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Determine file type
            filename = file.filename.lower()
            if filename.endswith('.csv'):
                file_type = 'csv'
            elif filename.endswith('.zip'):
                file_type = 'zip'
            else:
                return jsonify({'error': 'Unsupported file type. Please upload CSV or ZIP files.'}), 400
            
            # Get uploaded_by from form or default
            uploaded_by = request.form.get('uploaded_by', 'unknown')
            
            # Read file data
            file_data = file.read()
            
            # Create bulk upload job
            job = manager.create_bulk_upload_job(file.filename, file_type, uploaded_by)
            
            # Upload file to S3
            file_url = manager.upload_bulk_file(file_data, file.filename, file_type, uploaded_by)
            
            # Process the file based on type
            if file_type == 'csv':
                csv_data = file_data.decode('utf-8')
                result = manager.process_csv_bulk_upload(csv_data, job.id, uploaded_by)
            else:  # zip
                result = manager.process_zip_bulk_upload(file_data, job.id, uploaded_by)
            
            return jsonify({
                'job_id': job.id,
                'file_url': file_url,
                'result': result
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/bulk-upload/jobs', methods=['GET'])
    def list_bulk_upload_jobs():
        """List all bulk upload jobs with optional filtering."""
        try:
            status_filter = request.args.get('status', 'all')
            jobs = manager.list_bulk_upload_jobs(status_filter=status_filter)
            
            return jsonify([asdict(job) for job in jobs])
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/bulk-upload/jobs/<job_id>', methods=['GET'])
    def get_bulk_upload_job(job_id):
        """Get a specific bulk upload job."""
        try:
            job = manager.get_bulk_upload_job(job_id)
            if job:
                return jsonify(asdict(job))
            else:
                return jsonify({'error': 'Bulk upload job not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/bulk-upload/jobs/<job_id>/cancel', methods=['POST'])
    def cancel_bulk_upload_job(job_id):
        """Cancel a bulk upload job."""
        try:
            success = manager.update_bulk_upload_job(job_id, {'status': 'cancelled'})
            if success:
                return jsonify({'message': 'Bulk upload job cancelled successfully'})
            else:
                return jsonify({'error': 'Bulk upload job not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/bulk-upload/template', methods=['GET'])
    def get_bulk_upload_template():
        """Get a CSV template for bulk upload."""
        try:
            # Create CSV template with all required fields
            template_data = [
                {
                    'gloss': 'HELLO',
                    'english': 'Hello',
                    'handshape': 'B',
                    'location': 'neutral space',
                    'movement': 'wave',
                    'palm_orientation': 'palm forward',
                    'dominant_hand': 'B',
                    'non_dominant_hand': 'B',
                    'video_filename': 'hello.mp4',
                    'sign_type': 'isolated_sign',
                    'frequency': 100,
                    'age_of_acquisition': 2.5,
                    'iconicity': 0.8,
                    'lexical_class': 'interjection',
                    'tags': 'greeting,common',
                    'notes': 'Common greeting sign',
                    'uploaded_by': 'data_analyst'
                },
                {
                    'gloss': 'THANK-YOU',
                    'english': 'Thank you',
                    'handshape': 'A',
                    'location': 'chin',
                    'movement': 'forward',
                    'palm_orientation': 'palm in',
                    'dominant_hand': 'A',
                    'non_dominant_hand': 'B',
                    'video_filename': 'thank_you.mp4',
                    'sign_type': 'isolated_sign',
                    'frequency': 85,
                    'age_of_acquisition': 2.0,
                    'iconicity': 0.6,
                    'lexical_class': 'interjection',
                    'tags': 'politeness,common',
                    'notes': 'Polite expression of gratitude',
                    'uploaded_by': 'data_analyst'
                },
                {
                    'gloss': 'A-B-C',
                    'english': 'Alphabet',
                    'handshape': 'A',
                    'location': 'neutral space',
                    'movement': 'fingerspelling',
                    'palm_orientation': 'palm forward',
                    'dominant_hand': 'A',
                    'non_dominant_hand': 'B',
                    'video_filename': 'fingerspelling_abc.mp4',
                    'sign_type': 'fingerspelling',
                    'frequency': 75,
                    'age_of_acquisition': 3.0,
                    'iconicity': 0.3,
                    'lexical_class': 'noun',
                    'tags': 'alphabet,education',
                    'notes': 'Fingerspelling example',
                    'uploaded_by': 'data_analyst'
                }
            ]
            
            # Convert to CSV
            output = io.StringIO()
            if template_data:
                writer = csv.DictWriter(output, fieldnames=template_data[0].keys())
                writer.writeheader()
                writer.writerows(template_data)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=asl_lex_bulk_upload_template.csv'}
            )
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/signs/<sign_id>', methods=['GET'])
    def get_asl_lex_sign(sign_id):
        """Get a specific ASL-LEX sign."""
        try:
            sign = manager.get_sign(sign_id)
            if sign:
                return jsonify(asdict(sign))
            else:
                return jsonify({'error': 'Sign not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/signs', methods=['POST'])
    def create_asl_lex_sign():
        """Create a new ASL-LEX sign."""
        try:
            sign_data = request.json
            sign = manager.create_sign(sign_data)
            return jsonify(asdict(sign)), 201
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/signs/<sign_id>', methods=['PUT'])
    def update_asl_lex_sign(sign_id):
        """Update an ASL-LEX sign."""
        try:
            updates = request.json
            sign = manager.update_sign(sign_id, updates)
            if sign:
                return jsonify(asdict(sign))
            else:
                return jsonify({'error': 'Sign not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/signs/<sign_id>', methods=['DELETE'])
    def delete_asl_lex_sign(sign_id):
        """Delete an ASL-LEX sign."""
        try:
            success = manager.delete_sign(sign_id)
            if success:
                return jsonify({'message': 'Sign deleted successfully'})
            else:
                return jsonify({'error': 'Sign not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/signs/<sign_id>/validate', methods=['POST'])
    def validate_asl_lex_sign(sign_id):
        """Validate an ASL-LEX sign."""
        try:
            data = request.json
            validator = data.get('validator', 'unknown')
            is_valid = data.get('is_valid', False)
            notes = data.get('notes', '')
            
            success = manager.validate_sign(sign_id, validator, is_valid, notes)
            if success:
                return jsonify({'message': 'Sign validated successfully'})
            else:
                return jsonify({'error': 'Sign not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/statistics', methods=['GET'])
    def get_asl_lex_statistics():
        """Get ASL-LEX database statistics."""
        try:
            stats = manager.get_sign_statistics()
            return jsonify(stats)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/export', methods=['GET'])
    def export_asl_lex_signs():
        """Export ASL-LEX signs."""
        try:
            format = request.args.get('format', 'json')
            data = manager.export_signs(format)
            
            if format == 'json':
                return Response(data, mimetype='application/json')
            elif format == 'csv':
                return Response(data, mimetype='text/csv')
            else:
                return jsonify({'error': 'Unsupported format'}), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/import', methods=['POST'])
    def import_asl_lex_signs():
        """Import ASL-LEX signs."""
        try:
            format = request.args.get('format', 'json')
            data = request.get_data(as_text=True)
            
            imported_ids = manager.import_signs(data, format)
            return jsonify({
                'message': f'Imported {len(imported_ids)} signs',
                'imported_ids': imported_ids
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/signs/<sign_id>/video', methods=['POST'])
    def upload_sign_video(sign_id):
        """Upload a video for an ASL-LEX sign."""
        try:
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'No video file selected'}), 400
            
            # Read video data
            video_data = video_file.read()
            
            # Upload to S3
            video_url = manager.upload_sign_video(video_data, video_file.filename, sign_id)
            
            # Update sign with video URL
            manager.update_sign(sign_id, {'video_url': video_url})
            
            return jsonify({'video_url': video_url})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/upload-video-with-metadata', methods=['POST'])
    def upload_video_with_metadata():
        """Upload a video with metadata and sign type classification."""
        try:
            # Check for required files and data
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'No video file selected'}), 400
            
            # Get form data
            gloss = request.form.get('gloss', '').strip()
            english = request.form.get('english', '').strip()
            sign_type = request.form.get('sign_type', 'isolated_sign')
            uploaded_by = request.form.get('uploaded_by', 'unknown')
            
            if not gloss or not english:
                return jsonify({'error': 'Gloss and English translation are required'}), 400
            
            # Validate sign type
            valid_sign_types = [
                'isolated_sign',
                'continuous_signing', 
                'fingerspelling',
                'classifier',
                'compound_sign',
                'inflected_sign',
                'directional_sign',
                'spatial_sign',
                'temporal_sign',
                'manner_sign',
                'number_sign',
                'question_sign',
                'negation_sign',
                'modality_sign',
                'other'
            ]
            
            if sign_type not in valid_sign_types:
                return jsonify({'error': f'Invalid sign type. Must be one of: {", ".join(valid_sign_types)}'}), 400
            
            # Read video data
            video_data = video_file.read()
            
            # Create sign data
            sign_data = {
                'gloss': gloss,
                'english': english,
                'sign_type': sign_type,
                'handshape': request.form.get('handshape', ''),
                'location': request.form.get('location', ''),
                'movement': request.form.get('movement', ''),
                'palm_orientation': request.form.get('palm_orientation', ''),
                'dominant_hand': request.form.get('dominant_hand', ''),
                'non_dominant_hand': request.form.get('non_dominant_hand', ''),
                'frequency': int(request.form.get('frequency', 0)),
                'age_of_acquisition': float(request.form.get('age_of_acquisition', 0)),
                'iconicity': float(request.form.get('iconicity', 0)),
                'lexical_class': request.form.get('lexical_class', ''),
                'tags': request.form.get('tags', '').split(',') if request.form.get('tags') else [],
                'notes': request.form.get('notes', ''),
                'uploaded_by': uploaded_by,
                'confidence_score': float(request.form.get('confidence_score', 0.8))
            }
            
            # Create the sign first
            sign = manager.create_sign(sign_data)
            
            # Upload video and link it to the sign
            video_url = manager.upload_sign_video(video_data, video_file.filename, sign.id)
            manager.update_sign(sign.id, {'video_url': video_url})
            
            return jsonify({
                'success': True,
                'sign_id': sign.id,
                'video_url': video_url,
                'message': f'Sign "{gloss}" uploaded successfully as {sign_type}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/sign-types', methods=['GET'])
    def get_sign_types():
        """Get list of valid sign types for classification."""
        try:
            sign_types = [
                {
                    'value': 'isolated_sign',
                    'label': 'Isolated Sign',
                    'description': 'Single sign performed in isolation'
                },
                {
                    'value': 'continuous_signing',
                    'label': 'Continuous Signing',
                    'description': 'Sign within continuous signing context'
                },
                {
                    'value': 'fingerspelling',
                    'label': 'Fingerspelling',
                    'description': 'Manual alphabet or fingerspelling'
                },
                {
                    'value': 'classifier',
                    'label': 'Classifier',
                    'description': 'Classifier construction or classifier predicate'
                },
                {
                    'value': 'compound_sign',
                    'label': 'Compound Sign',
                    'description': 'Compound sign combining multiple elements'
                },
                {
                    'value': 'inflected_sign',
                    'label': 'Inflected Sign',
                    'description': 'Sign with grammatical inflection'
                },
                {
                    'value': 'directional_sign',
                    'label': 'Directional Sign',
                    'description': 'Sign with directional movement'
                },
                {
                    'value': 'spatial_sign',
                    'label': 'Spatial Sign',
                    'description': 'Sign involving spatial relationships'
                },
                {
                    'value': 'temporal_sign',
                    'label': 'Temporal Sign',
                    'description': 'Sign indicating time or temporal aspect'
                },
                {
                    'value': 'manner_sign',
                    'label': 'Manner Sign',
                    'description': 'Sign indicating manner of action'
                },
                {
                    'value': 'number_sign',
                    'label': 'Number Sign',
                    'description': 'Number or numeral sign'
                },
                {
                    'value': 'question_sign',
                    'label': 'Question Sign',
                    'description': 'Question or interrogative sign'
                },
                {
                    'value': 'negation_sign',
                    'label': 'Negation Sign',
                    'description': 'Negative or negation sign'
                },
                {
                    'value': 'modality_sign',
                    'label': 'Modality Sign',
                    'description': 'Sign indicating modality or mood'
                },
                {
                    'value': 'other',
                    'label': 'Other',
                    'description': 'Other type of sign not listed above'
                }
            ]
            
            return jsonify(sign_types)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/sign-types', methods=['POST'])
    def add_custom_sign_type():
        """Add a custom sign type for classification."""
        try:
            data = request.json
            custom_type = data.get('custom_type', '').strip()
            description = data.get('description', '').strip()
            
            if not custom_type:
                return jsonify({'error': 'Custom sign type is required'}), 400
            
            # Validate ASL-specific naming conventions
            if not custom_type.replace('_', '').replace('-', '').isalnum():
                return jsonify({'error': 'Sign type must contain only letters, numbers, underscores, and hyphens'}), 400
            
            # Check if custom type already exists
            existing_types = [
                'isolated_sign', 'continuous_signing', 'fingerspelling', 'classifier',
                'compound_sign', 'inflected_sign', 'directional_sign', 'spatial_sign',
                'temporal_sign', 'manner_sign', 'number_sign', 'question_sign',
                'negation_sign', 'modality_sign', 'other'
            ]
            
            if custom_type in existing_types:
                return jsonify({'error': 'Sign type already exists'}), 400
            
            # Store custom sign type in DynamoDB
            custom_type_item = {
                'id': f'custom_type_{custom_type}',
                'value': custom_type,
                'label': custom_type.replace('_', ' ').title(),
                'description': description,
                'is_custom': True,
                'created_by': data.get('created_by', 'unknown'),
                'created_at': datetime.utcnow().isoformat(),
                'timestamp': datetime.utcnow().isoformat(),
                'content_type': 'custom_sign_type'
            }
            
            manager.table.put_item(Item=custom_type_item)
            
            return jsonify({
                'success': True,
                'custom_type': custom_type_item,
                'message': f'Custom sign type "{custom_type}" added successfully'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/sign-types/custom', methods=['GET'])
    def get_custom_sign_types():
        """Get list of custom sign types."""
        try:
            response = manager.table.scan(
                FilterExpression='content_type = :content_type',
                ExpressionAttributeValues={
                    ':content_type': 'custom_sign_type'
                }
            )
            
            custom_types = []
            for item in response['Items']:
                custom_types.append({
                    'value': item['value'],
                    'label': item['label'],
                    'description': item.get('description', ''),
                    'is_custom': True,
                    'created_by': item.get('created_by', 'unknown'),
                    'created_at': item.get('created_at', '')
                })
            
            return jsonify(custom_types)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/signs/batch-update-type', methods=['POST'])
    def batch_update_sign_types():
        """Update sign types for multiple signs at once."""
        try:
            data = request.json
            sign_ids = data.get('sign_ids', [])
            new_sign_type = data.get('sign_type', '')
            updated_by = data.get('updated_by', 'unknown')
            
            if not sign_ids:
                return jsonify({'error': 'No sign IDs provided'}), 400
            
            if not new_sign_type:
                return jsonify({'error': 'New sign type is required'}), 400
            
            # Validate sign type
            valid_types = [
                'isolated_sign', 'continuous_signing', 'fingerspelling', 'classifier',
                'compound_sign', 'inflected_sign', 'directional_sign', 'spatial_sign',
                'temporal_sign', 'manner_sign', 'number_sign', 'question_sign',
                'negation_sign', 'modality_sign', 'other'
            ]
            
            # Check custom types
            custom_response = manager.table.scan(
                FilterExpression='content_type = :content_type',
                ExpressionAttributeValues={
                    ':content_type': 'custom_sign_type'
                }
            )
            
            custom_types = [item['value'] for item in custom_response['Items']]
            all_valid_types = valid_types + custom_types
            
            if new_sign_type not in all_valid_types:
                return jsonify({'error': f'Invalid sign type: {new_sign_type}'}), 400
            
            # Update each sign
            updated_count = 0
            failed_signs = []
            
            for sign_id in sign_ids:
                try:
                    # Get the sign
                    sign = manager.get_sign(sign_id)
                    if sign:
                        # Update sign type
                        manager.update_sign(sign_id, {
                            'sign_type': new_sign_type,
                            'updated_by': updated_by,
                            'updated_at': datetime.utcnow().isoformat()
                        })
                        updated_count += 1
                    else:
                        failed_signs.append(sign_id)
                except Exception as e:
                    failed_signs.append(sign_id)
                    logger.error(f"Error updating sign {sign_id}: {str(e)}")
            
            return jsonify({
                'success': True,
                'updated_count': updated_count,
                'failed_signs': failed_signs,
                'message': f'Updated {updated_count} signs to type "{new_sign_type}"'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/validate-asl-sign', methods=['POST'])
    def validate_asl_sign():
        """Validate ASL-specific sign characteristics."""
        try:
            data = request.json
            gloss = data.get('gloss', '').strip()
            handshape = data.get('handshape', '').strip()
            location = data.get('location', '').strip()
            movement = data.get('movement', '').strip()
            sign_type = data.get('sign_type', '').strip()
            
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'suggestions': []
            }
            
            # ASL-specific validation rules
            
            # 1. Gloss validation
            if gloss:
                if not gloss.isupper():
                    validation_results['warnings'].append('ASL glosses are typically written in UPPERCASE')
                
                if len(gloss) < 2:
                    validation_results['warnings'].append('Gloss seems too short for a typical ASL sign')
                
                # Check for common ASL gloss patterns
                if '-' in gloss and len(gloss.split('-')) > 3:
                    validation_results['warnings'].append('Complex gloss detected - consider if this is a compound sign')
            
            # 2. Handshape validation
            if handshape:
                valid_handshapes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                if handshape not in valid_handshapes:
                    validation_results['errors'].append(f'Invalid handshape "{handshape}". Valid handshapes: {", ".join(valid_handshapes)}')
            
            # 3. Location validation
            if location:
                valid_locations = [
                    'neutral space', 'head', 'face', 'chest', 'waist', 'chin', 
                    'forehead', 'nose', 'mouth', 'ear', 'eye', 'cheek', 
                    'shoulder', 'arm', 'hand', 'leg', 'foot'
                ]
                if location not in valid_locations:
                    validation_results['warnings'].append(f'Unusual location "{location}". Common ASL locations: {", ".join(valid_locations)}')
            
            # 4. Sign type validation
            if sign_type:
                if sign_type == 'fingerspelling':
                    if not gloss or len(gloss) > 1:
                        validation_results['warnings'].append('Fingerspelling typically represents single letters or short words')
                
                if sign_type == 'number_sign':
                    if not gloss.isdigit() and not any(word.isdigit() for word in gloss.split()):
                        validation_results['warnings'].append('Number signs typically have numeric glosses')
                
                if sign_type == 'question_sign':
                    if not any(q_word in gloss.upper() for q_word in ['WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW', 'WHICH']):
                        validation_results['warnings'].append('Question signs typically contain question words')
            
            # 5. Movement validation
            if movement:
                common_movements = ['wave', 'forward', 'backward', 'up', 'down', 'side', 'circle', 'straight', 'curved']
                if not any(m in movement.lower() for m in common_movements):
                    validation_results['suggestions'].append('Consider standardizing movement description')
            
            # 6. ASL-specific suggestions
            if gloss and 'HELLO' in gloss.upper():
                validation_results['suggestions'].append('Consider sign type "isolated_sign" for greeting signs')
            
            if gloss and any(word in gloss.upper() for word in ['THANK', 'SORRY', 'PLEASE']):
                validation_results['suggestions'].append('Consider sign type "interjection" for polite expressions')
            
            # Update validation status
            if validation_results['errors']:
                validation_results['is_valid'] = False
            
            return jsonify(validation_results)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/asl-lex/analytics/sign-types', methods=['GET'])
    def get_sign_type_analytics():
        """Get analytics on sign type distribution."""
        try:
            signs = manager.list_signs()
            
            # Count by sign type
            sign_type_counts = {}
            sign_type_details = {}
            
            for sign in signs:
                sign_type = sign.sign_type or 'unknown'
                if sign_type not in sign_type_counts:
                    sign_type_counts[sign_type] = 0
                    sign_type_details[sign_type] = {
                        'count': 0,
                        'examples': [],
                        'avg_confidence': 0,
                        'validation_status': {'validated': 0, 'unvalidated': 0, 'needs_review': 0}
                    }
                
                sign_type_counts[sign_type] += 1
                sign_type_details[sign_type]['count'] += 1
                
                # Add example (limit to 5)
                if len(sign_type_details[sign_type]['examples']) < 5:
                    sign_type_details[sign_type]['examples'].append({
                        'gloss': sign.gloss,
                        'english': sign.english,
                        'id': sign.id
                    })
                
                # Track validation status
                validation_status = sign.validation_status or 'unvalidated'
                sign_type_details[sign_type]['validation_status'][validation_status] += 1
            
            # Calculate averages and percentages
            total_signs = len(signs)
            sign_type_percentages = {}
            
            for sign_type, count in sign_type_counts.items():
                percentage = (count / total_signs * 100) if total_signs > 0 else 0
                sign_type_percentages[sign_type] = round(percentage, 2)
                
                # Calculate average confidence for this sign type
                type_signs = [s for s in signs if s.sign_type == sign_type]
                if type_signs:
                    avg_confidence = sum(s.confidence_score for s in type_signs) / len(type_signs)
                    sign_type_details[sign_type]['avg_confidence'] = round(avg_confidence, 3)
            
            # Get top sign types
            top_sign_types = sorted(sign_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Get recent trends (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_signs = [s for s in signs if datetime.fromisoformat(s.uploaded_at) > thirty_days_ago]
            
            recent_sign_type_counts = {}
            for sign in recent_signs:
                sign_type = sign.sign_type or 'unknown'
                recent_sign_type_counts[sign_type] = recent_sign_type_counts.get(sign_type, 0) + 1
            
            return jsonify({
                'total_signs': total_signs,
                'sign_type_counts': sign_type_counts,
                'sign_type_percentages': sign_type_percentages,
                'sign_type_details': sign_type_details,
                'top_sign_types': top_sign_types,
                'recent_trends': recent_sign_type_counts,
                'custom_types': manager.get_custom_sign_types()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Test the ASL-LEX data manager
    manager = ASLLexDataManager()
    
    # Test creating a sign
    test_sign_data = {
        'gloss': 'TEST',
        'english': 'Test',
        'handshape': 'A',
        'location': 'neutral space',
        'movement': 'test',
        'palm_orientation': 'palm forward',
        'dominant_hand': 'A',
        'non_dominant_hand': 'B',
        'video_url': '',
        'frequency': 50,
        'age_of_acquisition': 3.0,
        'iconicity': 0.5,
        'lexical_class': 'noun',
        'tags': ['test', 'example'],
        'notes': 'This is a test sign',
        'uploaded_by': 'test_user'
    }
    
    try:
        sign = manager.create_sign(test_sign_data)
        print(f"Created test sign: {sign.id}")
        
        # Test listing signs
        signs = manager.list_signs()
        print(f"Found {len(signs)} signs")
        
        # Test statistics
        stats = manager.get_sign_statistics()
        print(f"Statistics: {stats}")
        
    except Exception as e:
        print(f"Error testing ASL-LEX manager: {str(e)}") 