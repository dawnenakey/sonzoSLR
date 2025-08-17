"""
Video-Text Linking Service for Epic 3: Enhanced Video Workspace

This service integrates existing video infrastructure with Epic 2 text corpora,
enabling unified video-text annotation, search, and export capabilities.
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

# Import existing services
from text_corpus_service import TextCorpusService
from auth_service import AuthService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoTextLink:
    """Represents a link between video segment and text segment."""
    id: str
    video_id: str
    video_segment_id: str
    corpus_id: str
    text_segment_id: str
    link_type: str  # annotation, reference, translation, etc.
    confidence_score: float
    created_by: str
    created_at: str
    updated_at: str
    status: str  # draft, validated, approved, rejected
    metadata: Dict[str, Any]
    notes: str

@dataclass
class VideoTextAnnotation:
    """Enhanced annotation combining video and text data."""
    id: str
    video_id: str
    start_time: float
    end_time: float
    duration: float
    annotation_type: str  # sign_unit, segment, break, transition
    text_content: str
    linked_corpus_id: Optional[str]
    linked_text_segment_id: Optional[str]
    confidence_score: float
    hand_features: Dict[str, Any]
    spatial_features: Dict[str, Any]
    temporal_features: Dict[str, Any]
    created_by: str
    created_at: str
    updated_at: str
    status: str
    validation_notes: str
    tags: List[str]

@dataclass
class UnifiedSearchResult:
    """Result from cross-media search (video + text)."""
    result_type: str  # video, text, or combined
    video_id: Optional[str]
    text_corpus_id: Optional[str]
    text_segment_id: Optional[str]
    video_segment_id: Optional[str]
    content: str
    confidence_score: float
    metadata: Dict[str, Any]
    timestamp: str
    relevance_score: float

@dataclass
class VideoTextExport:
    """Export job for combined video-text data."""
    id: str
    video_id: str
    corpus_id: Optional[str]
    export_format: str  # json, csv, txt, combined
    status: str  # pending, processing, completed, failed
    created_by: str
    created_at: str
    completed_at: Optional[str]
    download_url: Optional[str]
    error_message: Optional[str]
    export_metadata: Dict[str, Any]

class VideoTextLinkingService:
    """Service for linking video content with text corpora."""
    
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table_prefix = os.getenv('DYNAMODB_TABLE_PREFIX', 'spokhand')
        
        # Initialize tables
        self.video_text_links_table = self.dynamodb.Table(f'{self.table_prefix}-video-text-links')
        self.video_text_annotations_table = self.dynamodb.Table(f'{self.table_prefix}-video-text-annotations')
        self.video_text_exports_table = self.dynamodb.Table(f'{self.table_prefix}-video-text-exports')
        
        # Initialize existing services
        self.text_corpus_service = TextCorpusService()
        self.auth_service = AuthService()
        
    def create_video_text_link(self, video_id: str, video_segment_id: str,
                              corpus_id: str, text_segment_id: str,
                              link_type: str, created_by: str,
                              confidence_score: float = 0.0,
                              metadata: Dict[str, Any] = None,
                              notes: str = "") -> VideoTextLink:
        """Create a link between video segment and text segment."""
        try:
            link_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            # Verify text segment exists
            text_segment = self.text_corpus_service.get_segment(text_segment_id)
            if not text_segment:
                raise ValueError(f"Text segment {text_segment_id} not found")
            
            # Verify corpus exists
            corpus = self.text_corpus_service.get_corpus(corpus_id)
            if not corpus:
                raise ValueError(f"Corpus {corpus_id} not found")
            
            link = VideoTextLink(
                id=link_id,
                video_id=video_id,
                video_segment_id=video_segment_id,
                corpus_id=corpus_id,
                text_segment_id=text_segment_id,
                link_type=link_type,
                confidence_score=confidence_score,
                created_by=created_by,
                created_at=now,
                updated_at=now,
                status='draft',
                metadata=metadata or {},
                notes=notes
            )
            
            # Store in DynamoDB
            self.video_text_links_table.put_item(Item=asdict(link))
            
            logger.info(f"Created video-text link {link_id} between video {video_id} and text {text_segment_id}")
            return link
            
        except Exception as e:
            logger.error(f"Error creating video-text link: {e}")
            raise
    
    def get_video_text_links(self, video_id: str = None, corpus_id: str = None,
                            link_type: str = None) -> List[VideoTextLink]:
        """Get video-text links with optional filtering."""
        try:
            scan_kwargs = {}
            
            if video_id:
                scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('video_id').eq(video_id)
            
            if corpus_id:
                if 'FilterExpression' in scan_kwargs:
                    scan_kwargs['FilterExpression'] &= boto3.dynamodb.conditions.Attr('corpus_id').eq(corpus_id)
                else:
                    scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('corpus_id').eq(corpus_id)
            
            if link_type:
                if 'FilterExpression' in scan_kwargs:
                    scan_kwargs['FilterExpression'] &= boto3.dynamodb.conditions.Attr('link_type').eq(link_type)
                else:
                    scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('link_type').eq(link_type)
            
            response = self.video_text_links_table.scan(**scan_kwargs)
            links = [VideoTextLink(**item) for item in response.get('Items', [])]
            
            # Sort by creation date
            links.sort(key=lambda x: x.created_at, reverse=True)
            
            return links
            
        except Exception as e:
            logger.error(f"Error getting video-text links: {e}")
            raise
    
    def update_video_text_link(self, link_id: str, updates: Dict[str, Any],
                              updated_by: str) -> Optional[VideoTextLink]:
        """Update a video-text link."""
        try:
            # Get current link
            response = self.video_text_links_table.get_item(Key={'id': link_id})
            if 'Item' not in response:
                return None
            
            current_link = VideoTextLink(**response['Item'])
            
            # Update fields
            for key, value in updates.items():
                if hasattr(current_link, key) and key not in ['id', 'video_id', 'corpus_id']:
                    setattr(current_link, key, value)
            
            current_link.updated_at = datetime.utcnow().isoformat()
            
            # Update in DynamoDB
            update_expression = "SET "
            expression_values = {}
            
            for key, value in updates.items():
                if key in ['link_type', 'confidence_score', 'status', 'metadata', 'notes']:
                    update_expression += f"#{key} = :{key}, "
                    expression_values[f":{key}"] = value
            
            update_expression += "#updated_at = :updated_at"
            expression_values[":updated_at"] = current_link.updated_at
            
            expression_names = {f"#{key}": key for key in updates.keys()}
            expression_names["#updated_at"] = "updated_at"
            
            self.video_text_links_table.update_item(
                Key={'id': link_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_names,
                ExpressionAttributeValues=expression_values
            )
            
            logger.info(f"Updated video-text link {link_id} by {updated_by}")
            return self.get_video_text_link(link_id)
            
        except Exception as e:
            logger.error(f"Error updating video-text link {link_id}: {e}")
            raise
    
    def get_video_text_link(self, link_id: str) -> Optional[VideoTextLink]:
        """Get a specific video-text link by ID."""
        try:
            response = self.video_text_links_table.get_item(Key={'id': link_id})
            if 'Item' in response:
                return VideoTextLink(**response['Item'])
            return None
        except Exception as e:
            logger.error(f"Error getting video-text link {link_id}: {e}")
            raise
    
    def delete_video_text_link(self, link_id: str, deleted_by: str) -> bool:
        """Delete a video-text link (soft delete)."""
        try:
            return self.update_video_text_link(
                link_id, 
                {'status': 'deleted'}, 
                deleted_by
            ) is not None
        except Exception as e:
            logger.error(f"Error deleting video-text link {link_id}: {e}")
            raise
    
    def create_video_text_annotation(self, video_id: str, start_time: float,
                                   end_time: float, annotation_type: str,
                                   text_content: str, created_by: str,
                                   linked_corpus_id: str = None,
                                   linked_text_segment_id: str = None,
                                   confidence_score: float = 0.0,
                                   hand_features: Dict[str, Any] = None,
                                   spatial_features: Dict[str, Any] = None,
                                   temporal_features: Dict[str, Any] = None,
                                   tags: List[str] = None) -> VideoTextAnnotation:
        """Create a video-text annotation combining video and text data."""
        try:
            annotation_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            duration = end_time - start_time
            
            annotation = VideoTextAnnotation(
                id=annotation_id,
                video_id=video_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                annotation_type=annotation_type,
                text_content=text_content,
                linked_corpus_id=linked_corpus_id,
                linked_text_segment_id=linked_text_segment_id,
                confidence_score=confidence_score,
                hand_features=hand_features or {},
                spatial_features=spatial_features or {},
                temporal_features=temporal_features or {},
                created_by=created_by,
                created_at=now,
                updated_at=now,
                status='draft',
                validation_notes='',
                tags=tags or []
            )
            
            # Store in DynamoDB
            self.video_text_annotations_table.put_item(Item=asdict(annotation))
            
            # If linked to text, create the link
            if linked_corpus_id and linked_text_segment_id:
                self.create_video_text_link(
                    video_id=video_id,
                    video_segment_id=annotation_id,
                    corpus_id=linked_corpus_id,
                    text_segment_id=linked_text_segment_id,
                    link_type='annotation',
                    created_by=created_by,
                    confidence_score=confidence_score
                )
            
            logger.info(f"Created video-text annotation {annotation_id} for video {video_id}")
            return annotation
            
        except Exception as e:
            logger.error(f"Error creating video-text annotation: {e}")
            raise
    
    def get_video_text_annotations(self, video_id: str = None,
                                 annotation_type: str = None,
                                 status: str = None) -> List[VideoTextAnnotation]:
        """Get video-text annotations with optional filtering."""
        try:
            scan_kwargs = {}
            
            if video_id:
                scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('video_id').eq(video_id)
            
            if annotation_type:
                if 'FilterExpression' in scan_kwargs:
                    scan_kwargs['FilterExpression'] &= boto3.dynamodb.conditions.Attr('annotation_type').eq(annotation_type)
                else:
                    scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('annotation_type').eq(annotation_type)
            
            if status:
                if 'FilterExpression' in scan_kwargs:
                    scan_kwargs['FilterExpression'] &= boto3.dynamodb.conditions.Attr('status').eq(status)
                else:
                    scan_kwargs['FilterExpression'] = boto3.dynamodb.conditions.Attr('status').eq(status)
            
            response = self.video_text_annotations_table.scan(**scan_kwargs)
            annotations = [VideoTextAnnotation(**item) for item in response.get('Items', [])]
            
            # Sort by start time
            annotations.sort(key=lambda x: x.start_time)
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error getting video-text annotations: {e}")
            raise
    
    def unified_search(self, query: str, search_type: str = 'combined',
                      video_id: str = None, corpus_id: str = None,
                      annotation_type: str = None) -> List[UnifiedSearchResult]:
        """Search across both video and text content."""
        try:
            results = []
            
            # Search in video-text annotations
            annotations = self.get_video_text_annotations(video_id=video_id)
            
            for annotation in annotations:
                # Text content search
                if query.lower() in annotation.text_content.lower():
                    results.append(UnifiedSearchResult(
                        result_type='combined',
                        video_id=annotation.video_id,
                        text_corpus_id=annotation.linked_corpus_id,
                        text_segment_id=annotation.linked_text_segment_id,
                        video_segment_id=annotation.id,
                        content=annotation.text_content,
                        confidence_score=annotation.confidence_score,
                        metadata={
                            'start_time': annotation.start_time,
                            'end_time': annotation.end_time,
                            'annotation_type': annotation.annotation_type,
                            'tags': annotation.tags
                        },
                        timestamp=annotation.created_at,
                        relevance_score=0.8  # Base relevance score
                    ))
                
                # Metadata search
                for tag in annotation.tags:
                    if query.lower() in tag.lower():
                        results.append(UnifiedSearchResult(
                            result_type='combined',
                            video_id=annotation.video_id,
                            text_corpus_id=annotation.linked_corpus_id,
                            text_segment_id=annotation.linked_text_segment_id,
                            video_segment_id=annotation.id,
                            content=f"Tag: {tag} - {annotation.text_content}",
                            confidence_score=annotation.confidence_score,
                            metadata={
                                'start_time': annotation.start_time,
                                'end_time': annotation.end_time,
                                'annotation_type': annotation.annotation_type,
                                'tags': annotation.tags
                            },
                            timestamp=annotation.created_at,
                            relevance_score=0.7
                        ))
                        break
            
            # Search in linked text corpora
            if search_type in ['combined', 'text']:
                links = self.get_video_text_links(video_id=video_id, corpus_id=corpus_id)
                
                for link in links:
                    # Get text segment content
                    text_segment = self.text_corpus_service.get_segment(link.text_segment_id)
                    if text_segment and query.lower() in text_segment.text.lower():
                        results.append(UnifiedSearchResult(
                            result_type='text',
                            video_id=link.video_id,
                            text_corpus_id=link.corpus_id,
                            text_segment_id=link.text_segment_id,
                            video_segment_id=link.video_segment_id,
                            content=text_segment.text,
                            confidence_score=link.confidence_score,
                            metadata={
                                'link_type': link.link_type,
                                'corpus_name': 'Linked Corpus',
                                'segment_type': text_segment.segment_type
                            },
                            timestamp=link.created_at,
                            relevance_score=0.9
                        ))
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in unified search: {e}")
            raise
    
    def create_video_text_export(self, video_id: str, export_format: str,
                                created_by: str, corpus_id: str = None,
                                export_metadata: Dict[str, Any] = None) -> VideoTextExport:
        """Create an export job for combined video-text data."""
        try:
            export_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            export_job = VideoTextExport(
                id=export_id,
                video_id=video_id,
                corpus_id=corpus_id,
                export_format=export_format,
                status='pending',
                created_by=created_by,
                created_at=now,
                completed_at=None,
                download_url=None,
                error_message=None,
                export_metadata=export_metadata or {}
            )
            
            # Store export job
            self.video_text_exports_table.put_item(Item=asdict(export_job))
            
            # Process export in background
            self._process_video_text_export(export_id, video_id, corpus_id, export_format)
            
            return export_job
            
        except Exception as e:
            logger.error(f"Error creating video-text export: {e}")
            raise
    
    def _process_video_text_export(self, export_id: str, video_id: str,
                                 corpus_id: str, export_format: str):
        """Process a video-text export job (background processing)."""
        try:
            # Update status to processing
            self.video_text_exports_table.update_item(
                Key={'id': export_id},
                UpdateExpression="SET #status = :status",
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': 'processing'}
            )
            
            # Get video annotations
            annotations = self.get_video_text_annotations(video_id=video_id)
            
            # Get linked text data if corpus specified
            text_data = []
            if corpus_id:
                links = self.get_video_text_links(video_id=video_id, corpus_id=corpus_id)
                for link in links:
                    text_segment = self.text_corpus_service.get_segment(link.text_segment_id)
                    if text_segment:
                        text_data.append({
                            'text_segment': asdict(text_segment),
                            'link': asdict(link)
                        })
            
            # Prepare export data
            export_data = {
                'video_id': video_id,
                'corpus_id': corpus_id,
                'annotations': [asdict(a) for a in annotations],
                'text_data': text_data,
                'exported_at': datetime.utcnow().isoformat(),
                'total_annotations': len(annotations),
                'total_text_segments': len(text_data)
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
                writer.writerow(['start_time', 'end_time', 'annotation_type', 'text_content', 'confidence_score'])
                for annotation in annotations:
                    writer.writerow([
                        annotation.start_time,
                        annotation.end_time,
                        annotation.annotation_type,
                        annotation.text_content,
                        annotation.confidence_score
                    ])
                export_content = output.getvalue()
            else:
                export_content = str(export_data)
            
            # Update export job as completed
            self.video_text_exports_table.update_item(
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
                    ':download_url': f'/api/video-text/exports/{export_id}/download'
                }
            )
            
            logger.info(f"Completed video-text export {export_id}")
            
        except Exception as e:
            logger.error(f"Error processing video-text export {export_id}: {e}")
            # Update export job as failed
            self.video_text_exports_table.update_item(
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
    
    def get_video_text_export_status(self, export_id: str) -> Optional[VideoTextExport]:
        """Get the status of a video-text export job."""
        try:
            response = self.video_text_exports_table.get_item(Key={'id': export_id})
            if 'Item' in response:
                return VideoTextExport(**response['Item'])
            return None
        except Exception as e:
            logger.error(f"Error getting export status {export_id}: {e}")
            raise
    
    def get_video_text_statistics(self, video_id: str = None,
                                corpus_id: str = None) -> Dict[str, Any]:
        """Get statistics for video-text integration."""
        try:
            # Get annotations
            annotations = self.get_video_text_annotations(video_id=video_id)
            
            # Get links
            links = self.get_video_text_links(video_id=video_id, corpus_id=corpus_id)
            
            # Calculate statistics
            total_annotations = len(annotations)
            total_links = len(links)
            
            # Annotation type distribution
            annotation_types = {}
            for annotation in annotations:
                ann_type = annotation.annotation_type
                annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1
            
            # Link type distribution
            link_types = {}
            for link in links:
                link_type = link.link_type
                link_types[link_type] = link_types.get(link_type, 0) + 1
            
            # Status distribution
            annotation_statuses = {}
            for annotation in annotations:
                status = annotation.status
                annotation_statuses[status] = annotation_statuses.get(status, 0) + 1
            
            return {
                'total_annotations': total_annotations,
                'total_links': total_links,
                'annotation_type_distribution': annotation_types,
                'link_type_distribution': link_types,
                'annotation_status_distribution': annotation_statuses,
                'linked_videos': len(set(link.video_id for link in links)),
                'linked_corpora': len(set(link.corpus_id for link in links))
            }
            
        except Exception as e:
            logger.error(f"Error getting video-text statistics: {e}")
            raise
