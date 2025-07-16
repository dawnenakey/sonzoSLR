"""
ASL-LEX Data Management Service

This service handles ASL-LEX sign data management including:
- Uploading sign videos to AWS S3
- Storing sign metadata in DynamoDB
- Managing sign validation and approval workflows
- Providing search and filtering capabilities
"""

import boto3
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import os
from botocore.exceptions import ClientError

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