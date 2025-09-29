"""
Lexicon Management Service for Epic 5: Lexicon Management

This service provides comprehensive lexicon management capabilities including:
- Sign CRUD operations
- Advanced search and filtering
- Batch operations
- Statistics and analytics
- Import/export functionality
- Validation workflows
"""

import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from flask import Flask, request, jsonify, send_file
from dataclasses import dataclass, asdict
import boto3
from botocore.exceptions import ClientError
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# AWS Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'spokhand-videos')
DYNAMODB_TABLE_PREFIX = os.getenv('DYNAMODB_TABLE_PREFIX', 'spokhand')

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

@dataclass
class LexiconSign:
    """Represents a sign in the lexicon with comprehensive metadata."""
    id: str
    gloss: str
    english: str
    handshape: str
    location: str
    movement: str
    palm_orientation: str
    dominant_hand: str
    non_dominant_hand: str
    video_url: Optional[str]
    sign_type: str
    frequency: int
    age_of_acquisition: float
    iconicity: float
    lexical_class: str
    tags: List[str]
    notes: str
    created_by: str
    created_at: str
    updated_at: str
    status: str  # pending, approved, rejected
    validation_status: str  # unvalidated, validated, needs_review
    confidence_score: float
    related_signs: List[str]
    etymology: str
    regional_variants: List[str]
    difficulty_level: int  # 1-5 scale
    usage_context: List[str]

@dataclass
class LexiconStatistics:
    """Statistics about the lexicon database."""
    total_signs: int
    approved_signs: int
    pending_signs: int
    rejected_signs: int
    validated_signs: int
    unvalidated_signs: int
    validation_rate: float
    handshape_distribution: Dict[str, int]
    location_distribution: Dict[str, int]
    sign_type_distribution: Dict[str, int]
    lexical_class_distribution: Dict[str, int]
    average_confidence: float
    recent_additions: int  # Last 30 days
    most_frequent_signs: List[Dict[str, Any]]

class LexiconManager:
    """Main class for managing the lexicon database."""
    
    def __init__(self):
        self.table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-lexicon-signs")
        self.statistics_table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-lexicon-statistics")
        
    def create_sign(self, sign_data: Dict[str, Any]) -> LexiconSign:
        """Create a new sign in the lexicon."""
        sign_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        sign = LexiconSign(
            id=sign_id,
            gloss=sign_data.get('gloss', ''),
            english=sign_data.get('english', ''),
            handshape=sign_data.get('handshape', ''),
            location=sign_data.get('location', ''),
            movement=sign_data.get('movement', ''),
            palm_orientation=sign_data.get('palm_orientation', ''),
            dominant_hand=sign_data.get('dominant_hand', ''),
            non_dominant_hand=sign_data.get('non_dominant_hand', ''),
            video_url=sign_data.get('video_url'),
            sign_type=sign_data.get('sign_type', 'isolated_sign'),
            frequency=sign_data.get('frequency', 0),
            age_of_acquisition=sign_data.get('age_of_acquisition', 0.0),
            iconicity=sign_data.get('iconicity', 0.0),
            lexical_class=sign_data.get('lexical_class', ''),
            tags=sign_data.get('tags', []),
            notes=sign_data.get('notes', ''),
            created_by=sign_data.get('created_by', 'system'),
            created_at=now,
            updated_at=now,
            status=sign_data.get('status', 'pending'),
            validation_status=sign_data.get('validation_status', 'unvalidated'),
            confidence_score=sign_data.get('confidence_score', 0.0),
            related_signs=sign_data.get('related_signs', []),
            etymology=sign_data.get('etymology', ''),
            regional_variants=sign_data.get('regional_variants', []),
            difficulty_level=sign_data.get('difficulty_level', 1),
            usage_context=sign_data.get('usage_context', [])
        )
        
        # Save to DynamoDB
        self.table.put_item(Item=asdict(sign))
        
        # Update statistics
        self._update_statistics()
        
        return sign
    
    def get_sign(self, sign_id: str) -> Optional[LexiconSign]:
        """Get a specific sign by ID."""
        try:
            response = self.table.get_item(Key={'id': sign_id})
            if 'Item' in response:
                return LexiconSign(**response['Item'])
            return None
        except Exception as e:
            logger.error(f"Error getting sign {sign_id}: {str(e)}")
            return None
    
    def update_sign(self, sign_id: str, updates: Dict[str, Any]) -> Optional[LexiconSign]:
        """Update an existing sign."""
        try:
            updates['updated_at'] = datetime.now().isoformat()
            
            # Build update expression
            update_expression = "SET " + ", ".join([f"{k} = :{k}" for k in updates.keys()])
            expression_values = {f":{k}": v for k, v in updates.items()}
            
            self.table.update_item(
                Key={'id': sign_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            
            # Return updated sign
            return self.get_sign(sign_id)
            
        except Exception as e:
            logger.error(f"Error updating sign {sign_id}: {str(e)}")
            return None
    
    def delete_sign(self, sign_id: str) -> bool:
        """Delete a sign from the lexicon."""
        try:
            self.table.delete_item(Key={'id': sign_id})
            self._update_statistics()
            return True
        except Exception as e:
            logger.error(f"Error deleting sign {sign_id}: {str(e)}")
            return False
    
    def list_signs(self, 
                   status_filter: str = 'all',
                   handshape_filter: str = 'all',
                   location_filter: str = 'all',
                   sign_type_filter: str = 'all',
                   search_term: str = '',
                   limit: int = 100,
                   offset: int = 0) -> List[LexiconSign]:
        """List signs with filtering and pagination."""
        try:
            # Build filter expressions
            filter_expressions = []
            expression_values = {}
            
            if status_filter != 'all':
                filter_expressions.append("status = :status")
                expression_values[':status'] = status_filter
            
            if handshape_filter != 'all':
                filter_expressions.append("handshape = :handshape")
                expression_values[':handshape'] = handshape_filter
            
            if location_filter != 'all':
                filter_expressions.append("location = :location")
                expression_values[':location'] = location_filter
            
            if sign_type_filter != 'all':
                filter_expressions.append("sign_type = :sign_type")
                expression_values[':sign_type'] = sign_type_filter
            
            if search_term:
                filter_expressions.append("(contains(gloss, :search) OR contains(english, :search))")
                expression_values[':search'] = search_term.lower()
            
            # Build query parameters
            query_params = {
                'Limit': limit,
                'ScanIndexForward': False  # Sort by created_at descending
            }
            
            if filter_expressions:
                query_params['FilterExpression'] = " AND ".join(filter_expressions)
                query_params['ExpressionAttributeValues'] = expression_values
            
            # Execute scan (DynamoDB doesn't support complex queries on GSI)
            response = self.table.scan(**query_params)
            items = response.get('Items', [])
            
            # Convert to LexiconSign objects
            signs = [LexiconSign(**item) for item in items]
            
            # Apply pagination
            if offset > 0:
                signs = signs[offset:]
            
            return signs
            
        except Exception as e:
            logger.error(f"Error listing signs: {str(e)}")
            return []
    
    def batch_update_signs(self, sign_ids: List[str], updates: Dict[str, Any]) -> int:
        """Update multiple signs in batch."""
        updated_count = 0
        
        for sign_id in sign_ids:
            if self.update_sign(sign_id, updates):
                updated_count += 1
        
        if updated_count > 0:
            self._update_statistics()
        
        return updated_count
    
    def get_statistics(self) -> LexiconStatistics:
        """Get comprehensive statistics about the lexicon."""
        try:
            # Try to get cached statistics first
            response = self.statistics_table.get_item(Key={'id': 'main'})
            if 'Item' in response:
                stats_data = response['Item']
                return LexiconStatistics(**stats_data)
            
            # If no cached stats, calculate them
            return self._calculate_statistics()
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return self._calculate_statistics()
    
    def _calculate_statistics(self) -> LexiconStatistics:
        """Calculate fresh statistics from the database."""
        try:
            # Get all signs
            response = self.table.scan()
            signs = [LexiconSign(**item) for item in response.get('Items', [])]
            
            total_signs = len(signs)
            approved_signs = len([s for s in signs if s.status == 'approved'])
            pending_signs = len([s for s in signs if s.status == 'pending'])
            rejected_signs = len([s for s in signs if s.status == 'rejected'])
            validated_signs = len([s for s in signs if s.validation_status == 'validated'])
            unvalidated_signs = len([s for s in signs if s.validation_status == 'unvalidated'])
            
            validation_rate = (validated_signs / total_signs * 100) if total_signs > 0 else 0
            
            # Distribution calculations
            handshape_dist = {}
            location_dist = {}
            sign_type_dist = {}
            lexical_class_dist = {}
            
            for sign in signs:
                handshape_dist[sign.handshape] = handshape_dist.get(sign.handshape, 0) + 1
                location_dist[sign.location] = location_dist.get(sign.location, 0) + 1
                sign_type_dist[sign.sign_type] = sign_type_dist.get(sign.sign_type, 0) + 1
                lexical_class_dist[sign.lexical_class] = lexical_class_dist.get(sign.lexical_class, 0) + 1
            
            # Average confidence
            avg_confidence = sum(s.confidence_score for s in signs) / total_signs if total_signs > 0 else 0
            
            # Recent additions (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_additions = len([s for s in signs if datetime.fromisoformat(s.created_at) > thirty_days_ago])
            
            # Most frequent signs
            most_frequent = sorted(signs, key=lambda s: s.frequency, reverse=True)[:10]
            most_frequent_data = [{'gloss': s.gloss, 'frequency': s.frequency} for s in most_frequent]
            
            stats = LexiconStatistics(
                total_signs=total_signs,
                approved_signs=approved_signs,
                pending_signs=pending_signs,
                rejected_signs=rejected_signs,
                validated_signs=validated_signs,
                unvalidated_signs=unvalidated_signs,
                validation_rate=validation_rate,
                handshape_distribution=handshape_dist,
                location_distribution=location_dist,
                sign_type_distribution=sign_type_dist,
                lexical_class_distribution=lexical_class_dist,
                average_confidence=avg_confidence,
                recent_additions=recent_additions,
                most_frequent_signs=most_frequent_data
            )
            
            # Cache the statistics
            self._cache_statistics(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return LexiconStatistics(
                total_signs=0, approved_signs=0, pending_signs=0, rejected_signs=0,
                validated_signs=0, unvalidated_signs=0, validation_rate=0,
                handshape_distribution={}, location_distribution={}, sign_type_distribution={},
                lexical_class_distribution={}, average_confidence=0, recent_additions=0,
                most_frequent_signs=[]
            )
    
    def _update_statistics(self):
        """Update cached statistics."""
        stats = self._calculate_statistics()
        self._cache_statistics(stats)
    
    def _cache_statistics(self, stats: LexiconStatistics):
        """Cache statistics in DynamoDB."""
        try:
            self.statistics_table.put_item(
                Item={
                    'id': 'main',
                    'data': asdict(stats),
                    'updated_at': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error caching statistics: {str(e)}")
    
    def export_lexicon(self, format: str = 'json') -> str:
        """Export the entire lexicon to a file."""
        try:
            signs = self.list_signs(limit=10000)  # Get all signs
            
            if format == 'json':
                data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_signs': len(signs),
                    'signs': [asdict(sign) for sign in signs]
                }
                return json.dumps(data, indent=2)
            elif format == 'csv':
                # Convert to CSV format
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                if signs:
                    writer.writerow(signs[0].__dict__.keys())
                    for sign in signs:
                        writer.writerow(sign.__dict__.values())
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting lexicon: {str(e)}")
            raise
    
    def import_lexicon(self, data: str, format: str = 'json') -> int:
        """Import signs from a file."""
        try:
            imported_count = 0
            
            if format == 'json':
                import_data = json.loads(data)
                signs_data = import_data.get('signs', [])
                
                for sign_data in signs_data:
                    self.create_sign(sign_data)
                    imported_count += 1
                    
            elif format == 'csv':
                import csv
                import io
                reader = csv.DictReader(io.StringIO(data))
                
                for row in reader:
                    self.create_sign(row)
                    imported_count += 1
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing lexicon: {str(e)}")
            raise

# Initialize lexicon manager
lexicon_manager = LexiconManager()

# API Routes
@app.route('/api/lexicon/signs', methods=['GET'])
def list_signs():
    """List signs with filtering."""
    try:
        status_filter = request.args.get('status', 'all')
        handshape_filter = request.args.get('handshape', 'all')
        location_filter = request.args.get('location', 'all')
        sign_type_filter = request.args.get('sign_type', 'all')
        search_term = request.args.get('search', '')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        signs = lexicon_manager.list_signs(
            status_filter=status_filter,
            handshape_filter=handshape_filter,
            location_filter=location_filter,
            sign_type_filter=sign_type_filter,
            search_term=search_term,
            limit=limit,
            offset=offset
        )
        
        return jsonify([asdict(sign) for sign in signs])
        
    except Exception as e:
        logger.error(f"Error listing signs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/signs', methods=['POST'])
def create_sign():
    """Create a new sign."""
    try:
        sign_data = request.get_json()
        sign = lexicon_manager.create_sign(sign_data)
        
        return jsonify(asdict(sign)), 201
        
    except Exception as e:
        logger.error(f"Error creating sign: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/signs/<sign_id>', methods=['GET'])
def get_sign(sign_id):
    """Get a specific sign."""
    try:
        sign = lexicon_manager.get_sign(sign_id)
        if not sign:
            return jsonify({'error': 'Sign not found'}), 404
        
        return jsonify(asdict(sign))
        
    except Exception as e:
        logger.error(f"Error getting sign: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/signs/<sign_id>', methods=['PUT'])
def update_sign(sign_id):
    """Update a sign."""
    try:
        updates = request.get_json()
        sign = lexicon_manager.update_sign(sign_id, updates)
        
        if not sign:
            return jsonify({'error': 'Sign not found'}), 404
        
        return jsonify(asdict(sign))
        
    except Exception as e:
        logger.error(f"Error updating sign: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/signs/<sign_id>', methods=['DELETE'])
def delete_sign(sign_id):
    """Delete a sign."""
    try:
        success = lexicon_manager.delete_sign(sign_id)
        if not success:
            return jsonify({'error': 'Sign not found'}), 404
        
        return jsonify({'message': 'Sign deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting sign: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/signs/batch-update', methods=['POST'])
def batch_update_signs():
    """Update multiple signs in batch."""
    try:
        data = request.get_json()
        sign_ids = data.get('sign_ids', [])
        updates = data.get('updates', {})
        
        updated_count = lexicon_manager.batch_update_signs(sign_ids, updates)
        
        return jsonify({
            'message': f'Updated {updated_count} signs',
            'updated_count': updated_count
        })
        
    except Exception as e:
        logger.error(f"Error batch updating signs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/statistics', methods=['GET'])
def get_statistics():
    """Get lexicon statistics."""
    try:
        stats = lexicon_manager.get_statistics()
        return jsonify(asdict(stats))
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/export', methods=['GET'])
def export_lexicon():
    """Export the lexicon."""
    try:
        format_type = request.args.get('format', 'json')
        data = lexicon_manager.export_lexicon(format_type)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format_type}', delete=False) as temp_file:
            temp_file.write(data)
            temp_file_path = temp_file.name
        
        return send_file(
            temp_file_path,
            as_attachment=True,
            download_name=f'lexicon_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format_type}',
            mimetype='application/json' if format_type == 'json' else 'text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting lexicon: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/import', methods=['POST'])
def import_lexicon():
    """Import signs to the lexicon."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        format_type = request.form.get('format', 'json')
        
        data = file.read().decode('utf-8')
        imported_count = lexicon_manager.import_lexicon(data, format_type)
        
        return jsonify({
            'message': f'Imported {imported_count} signs',
            'imported_count': imported_count
        })
        
    except Exception as e:
        logger.error(f"Error importing lexicon: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lexicon/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        stats = lexicon_manager.get_statistics()
        return jsonify({
            'status': 'healthy',
            'total_signs': stats.total_signs,
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'last_updated': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
