"""
Text Corpus Management Service for SPOKHAND SIGNCUT

This service handles text corpus creation, management, and integration with sign language data.
Provides the foundation for Epic 2: Text Corpus Management.
"""

import boto3
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from botocore.exceptions import ClientError
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextCorpus:
    """Represents a text corpus for sign language data."""
    id: str
    name: str
    description: str
    language: str  # ASL, BSL, etc.
    metadata: Dict[str, Any]
    created_by: str
    created_at: str
    updated_at: str
    status: str  # draft, active, archived, deleted
    total_segments: int
    validated_segments: int
    tags: List[str]
    version: str

@dataclass
class TextSegment:
    """Represents a text segment within a corpus."""
    id: str
    corpus_id: str
    text: str
    metadata: Dict[str, Any]
    position: int
    segment_type: str  # sentence, phrase, word, paragraph
    created_by: str
    created_at: str
    updated_at: str
    status: str  # draft, validated, approved, rejected
    validation_notes: str
    related_signs: List[str]  # List of sign IDs
    confidence_score: float

@dataclass
class CorpusExport:
    """Represents a corpus export job."""
    id: str
    corpus_id: str
    export_format: str  # json, csv, txt
    status: str  # pending, processing, completed, failed
    created_by: str
    created_at: str
    completed_at: Optional[str]
    download_url: Optional[str]
    error_message: Optional[str]

class TextCorpusService:
    """Manages text corpora and segments in AWS DynamoDB."""
    
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table_prefix = os.getenv('DYNAMODB_TABLE_PREFIX', 'spokhand')
        self.corpora_table = self.dynamodb.Table(f'{self.table_prefix}-text-corpora')
        self.segments_table = self.dynamodb.Table(f'{self.table_prefix}-text-segments')
        self.exports_table = self.dynamodb.Table(f'{self.table_prefix}-corpus-exports')
        
    def create_corpus(self, name: str, description: str, language: str, 
                     created_by: str, metadata: Dict[str, Any] = None, 
                     tags: List[str] = None) -> TextCorpus:
        """Create a new text corpus."""
        try:
            corpus_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            corpus = TextCorpus(
                id=corpus_id,
                name=name,
                description=description,
                language=language,
                metadata=metadata or {},
                created_by=created_by,
                created_at=now,
                updated_at=now,
                status='draft',
                total_segments=0,
                validated_segments=0,
                tags=tags or [],
                version='1.0.0'
            )
            
            # Store in DynamoDB
            self.corpora_table.put_item(Item=asdict(corpus))
            
            logger.info(f"Created corpus {corpus_id} by {created_by}")
            return corpus
            
        except Exception as e:
            logger.error(f"Error creating corpus: {e}")
            raise
    
    def get_corpus(self, corpus_id: str) -> Optional[TextCorpus]:
        """Retrieve a corpus by ID."""
        try:
            response = self.corpora_table.get_item(Key={'id': corpus_id})
            if 'Item' in response:
                return TextCorpus(**response['Item'])
            return None
        except Exception as e:
            logger.error(f"Error retrieving corpus {corpus_id}: {e}")
            raise
    
    def list_corpora(self, user_id: str = None, status: str = None, 
                    language: str = None) -> List[TextCorpus]:
        """List corpora with optional filtering."""
        try:
            # Start with scan
            scan_kwargs = {}
            
            if status:
                scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('status').eq(status)
            
            if language:
                if 'FilterExpression' in scan_kwargs:
                    scan_kwargs['FilterExpression'] &= boto3.dynamodb.conditions.Attr('language').eq(language)
                else:
                    scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('language').eq(language)
            
            response = self.corpora_table.scan(**scan_kwargs)
            corpora = [TextCorpus(**item) for item in response.get('Items', [])]
            
            # Filter by user if specified
            if user_id:
                corpora = [c for c in corpora if c.created_by == user_id]
            
            # Sort by creation date
            corpora.sort(key=lambda x: x.created_at, reverse=True)
            
            return corpora
            
        except Exception as e:
            logger.error(f"Error listing corpora: {e}")
            raise
    
    def update_corpus(self, corpus_id: str, updates: Dict[str, Any], 
                     updated_by: str) -> Optional[TextCorpus]:
        """Update corpus metadata."""
        try:
            # Get current corpus
            corpus = self.get_corpus(corpus_id)
            if not corpus:
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(corpus, key) and key not in ['id', 'created_by', 'created_at']:
                    setattr(corpus, key, value)
            
            corpus.updated_at = datetime.utcnow().isoformat()
            
            # Update in DynamoDB
            update_expression = "SET "
            expression_values = {}
            
            for key, value in updates.items():
                if key in ['name', 'description', 'language', 'metadata', 'status', 'tags', 'version']:
                    update_expression += f"#{key} = :{key}, "
                    expression_values[f":{key}"] = value
            
            update_expression += "#updated_at = :updated_at"
            expression_values[":updated_at"] = corpus.updated_at
            
            expression_names = {f"#{key}": key for key in updates.keys()}
            expression_names["#updated_at"] = "updated_at"
            
            self.corpora_table.update_item(
                Key={'id': corpus_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_names,
                ExpressionAttributeValues=expression_values
            )
            
            logger.info(f"Updated corpus {corpus_id} by {updated_by}")
            return self.get_corpus(corpus_id)
            
        except Exception as e:
            logger.error(f"Error updating corpus {corpus_id}: {e}")
            raise
    
    def delete_corpus(self, corpus_id: str, deleted_by: str) -> bool:
        """Soft delete a corpus (mark as deleted)."""
        try:
            # Soft delete - mark as deleted
            self.update_corpus(corpus_id, {'status': 'deleted'}, deleted_by)
            
            # Mark all segments as deleted
            segments = self.list_segments(corpus_id)
            for segment in segments:
                self.update_segment(segment.id, {'status': 'deleted'}, deleted_by)
            
            logger.info(f"Deleted corpus {corpus_id} by {deleted_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting corpus {corpus_id}: {e}")
            raise
    
    def add_text_segment(self, corpus_id: str, text: str, segment_type: str,
                        created_by: str, metadata: Dict[str, Any] = None,
                        related_signs: List[str] = None) -> TextSegment:
        """Add a new text segment to a corpus."""
        try:
            # Verify corpus exists
            corpus = self.get_corpus(corpus_id)
            if not corpus:
                raise ValueError(f"Corpus {corpus_id} not found")
            
            # Get next position
            existing_segments = self.list_segments(corpus_id)
            position = len(existing_segments) + 1
            
            segment_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            segment = TextSegment(
                id=segment_id,
                corpus_id=corpus_id,
                text=text,
                metadata=metadata or {},
                position=position,
                segment_type=segment_type,
                created_by=created_by,
                created_at=now,
                updated_at=now,
                status='draft',
                validation_notes='',
                related_signs=related_signs or [],
                confidence_score=0.0
            )
            
            # Store segment
            self.segments_table.put_item(Item=asdict(segment))
            
            # Update corpus segment count
            self.corpora_table.update_item(
                Key={'id': corpus_id},
                UpdateExpression="SET total_segments = total_segments + :inc, updated_at = :updated_at",
                ExpressionAttributeValues={
                    ':inc': 1,
                    ':updated_at': now
                }
            )
            
            logger.info(f"Added segment {segment_id} to corpus {corpus_id}")
            return segment
            
        except Exception as e:
            logger.error(f"Error adding segment to corpus {corpus_id}: {e}")
            raise
    
    def get_segment(self, segment_id: str) -> Optional[TextSegment]:
        """Retrieve a text segment by ID."""
        try:
            response = self.segments_table.get_item(Key={'id': segment_id})
            if 'Item' in response:
                return TextSegment(**response['Item'])
            return None
        except Exception as e:
            logger.error(f"Error retrieving segment {segment_id}: {e}")
            raise
    
    def list_segments(self, corpus_id: str, status: str = None) -> List[TextSegment]:
        """List segments in a corpus."""
        try:
            scan_kwargs = {
                'FilterExpression': boto3.dynamodb.conditions.Attr('corpus_id').eq(corpus_id)
            }
            
            if status:
                scan_kwargs['FilterExpression'] &= boto3.dynamodb.conditions.Attr('status').eq(status)
            
            response = self.segments_table.scan(**scan_kwargs)
            segments = [TextSegment(**item) for item in response.get('Items', [])]
            
            # Sort by position
            segments.sort(key=lambda x: x.position)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error listing segments for corpus {corpus_id}: {e}")
            raise
    
    def update_segment(self, segment_id: str, updates: Dict[str, Any], 
                      updated_by: str) -> Optional[TextSegment]:
        """Update a text segment."""
        try:
            # Get current segment
            segment = self.get_segment(segment_id)
            if not segment:
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(segment, key) and key not in ['id', 'corpus_id', 'created_by', 'created_at']:
                    setattr(segment, key, value)
            
            segment.updated_at = datetime.utcnow().isoformat()
            
            # Update in DynamoDB
            update_expression = "SET "
            expression_values = {}
            
            for key, value in updates.items():
                if key in ['text', 'metadata', 'segment_type', 'status', 'validation_notes', 'related_signs', 'confidence_score']:
                    update_expression += f"#{key} = :{key}, "
                    expression_values[f":{key}"] = value
            
            update_expression += "#updated_at = :updated_at"
            expression_values[":updated_at"] = segment.updated_at
            
            expression_names = {f"#{key}": key for key in updates.keys()}
            expression_names["#updated_at"] = "updated_at"
            
            self.segments_table.update_item(
                Key={'id': segment_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_names,
                ExpressionAttributeValues=expression_values
            )
            
            logger.info(f"Updated segment {segment_id} by {updated_by}")
            return self.get_segment(segment_id)
            
        except Exception as e:
            logger.error(f"Error updating segment {segment_id}: {e}")
            raise
    
    def delete_segment(self, segment_id: str, deleted_by: str) -> bool:
        """Soft delete a text segment."""
        try:
            segment = self.get_segment(segment_id)
            if not segment:
                return False
            
            # Soft delete
            self.update_segment(segment_id, {'status': 'deleted'}, deleted_by)
            
            # Update corpus segment count
            self.corpora_table.update_item(
                Key={'id': segment.corpus_id},
                UpdateExpression="SET total_segments = total_segments - :dec, updated_at = :updated_at",
                ExpressionAttributeValues={
                    ':dec': 1,
                    ':updated_at': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Deleted segment {segment_id} by {deleted_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting segment {segment_id}: {e}")
            raise
    
    def search_corpus(self, corpus_id: str, query: str, 
                     search_type: str = 'text') -> List[TextSegment]:
        """Search within a corpus."""
        try:
            segments = self.list_segments(corpus_id)
            
            if search_type == 'text':
                # Simple text search
                results = [s for s in segments if query.lower() in s.text.lower()]
            elif search_type == 'metadata':
                # Search in metadata
                results = []
                for segment in segments:
                    for key, value in segment.metadata.items():
                        if isinstance(value, str) and query.lower() in value.lower():
                            results.append(segment)
                            break
            else:
                results = segments
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching corpus {corpus_id}: {e}")
            raise
    
    def export_corpus(self, corpus_id: str, export_format: str, 
                     created_by: str) -> CorpusExport:
        """Create a corpus export job."""
        try:
            export_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            export_job = CorpusExport(
                id=export_id,
                corpus_id=corpus_id,
                export_format=export_format,
                status='pending',
                created_by=created_by,
                created_at=now,
                completed_at=None,
                download_url=None,
                error_message=None
            )
            
            # Store export job
            self.exports_table.put_item(Item=asdict(export_job))
            
            # Process export in background (simplified for now)
            self._process_export(export_id, corpus_id, export_format)
            
            return export_job
            
        except Exception as e:
            logger.error(f"Error creating export for corpus {corpus_id}: {e}")
            raise
    
    def _process_export(self, export_id: str, corpus_id: str, export_format: str):
        """Process an export job (simplified implementation)."""
        try:
            # Get corpus and segments
            corpus = self.get_corpus(corpus_id)
            segments = self.list_segments(corpus_id)
            
            # Update status to processing
            self.exports_table.update_item(
                Key={'id': export_id},
                UpdateExpression="SET #status = :status",
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': 'processing'}
            )
            
            # Generate export data
            export_data = {
                'corpus': asdict(corpus),
                'segments': [asdict(s) for s in segments],
                'exported_at': datetime.utcnow().isoformat(),
                'total_segments': len(segments)
            }
            
            # Convert to requested format
            if export_format == 'json':
                export_content = json.dumps(export_data, indent=2)
            elif export_format == 'csv':
                # Simplified CSV export
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(['position', 'text', 'segment_type', 'status', 'related_signs'])
                for segment in segments:
                    writer.writerow([
                        segment.position,
                        segment.text,
                        segment.segment_type,
                        segment.status,
                        ','.join(segment.related_signs)
                    ])
                export_content = output.getvalue()
            else:
                export_content = str(export_data)
            
            # Update export job as completed
            self.exports_table.update_item(
                Key={'id': export_id},
                UpdateExpression="SET #status = :status, #completed_at = :completed_at, #download_url = :download_url",
                ExpressionAttributeNames={
                    '#status': 'status',
                    '#completed_at': 'completed_at',
                    '#download_url': 'download_url'
                },
                ExpressionAttributeValues={
                    ':status': 'completed',
                    ':completed_at': datetime.utcnow().isoformat(),
                    ':download_url': f'/api/corpora/exports/{export_id}/download'
                }
            )
            
            logger.info(f"Completed export {export_id} for corpus {corpus_id}")
            
        except Exception as e:
            logger.error(f"Error processing export {export_id}: {e}")
            # Update export job as failed
            self.exports_table.update_item(
                Key={'id': export_id},
                UpdateExpression="SET #status = :status, #error_message = :error_message",
                ExpressionAttributeNames={
                    '#status': 'status',
                    '#error_message': 'error_message'
                },
                ExpressionAttributeValues={
                    ':status': 'failed',
                    ':error_message': str(e)
                }
            )
    
    def get_export_status(self, export_id: str) -> Optional[CorpusExport]:
        """Get the status of an export job."""
        try:
            response = self.exports_table.get_item(Key={'id': export_id})
            if 'Item' in response:
                return CorpusExport(**response['Item'])
            return None
        except Exception as e:
            logger.error(f"Error getting export status {export_id}: {e}")
            raise
    
    def list_exports(self, corpus_id: str = None, user_id: str = None) -> List[CorpusExport]:
        """List export jobs."""
        try:
            scan_kwargs = {}
            
            if corpus_id:
                scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('corpus_id').eq(corpus_id)
            
            if user_id:
                if 'FilterExpression' in scan_kwargs:
                    scan_kwargs['FilterExpression'] &= boto3.dynamodb.conditions.Attr('created_by').eq(user_id)
                else:
                    scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('created_by').eq(user_id)
            
            response = self.exports_table.scan(**scan_kwargs)
            exports = [CorpusExport(**item) for item in response.get('Items', [])]
            
            # Sort by creation date
            exports.sort(key=lambda x: x.created_at, reverse=True)
            
            return exports
            
        except Exception as e:
            logger.error(f"Error listing exports: {e}")
            raise
