"""
Unified Video-Text API for Epic 3: Enhanced Video Workspace

This API integrates video-text linking with Epic 1 authentication and Epic 2 text corpora,
providing unified endpoints for video-text annotation, search, and export.
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import os

# Import our services
from video_text_linking_service import VideoTextLinkingService
from text_corpus_service import TextCorpusService
from auth_service import AuthService
from auth_api import require_auth, get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
video_text_service = VideoTextLinkingService()
text_corpus_service = TextCorpusService()
auth_service = AuthService()

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')

@app.before_request
def before_request():
    """Set up request context."""
    g.start_time = datetime.utcnow()

@app.after_request
def after_request(response):
    """Log request details."""
    if hasattr(g, 'start_time'):
        duration = (datetime.utcnow() - g.start_time).total_seconds()
        logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'video-text-unified-api',
        'epic': 'Epic 3 - Enhanced Video Workspace',
        'timestamp': datetime.utcnow().isoformat()
    })

# ============================================================================
# VIDEO-TEXT LINKING ENDPOINTS
# ============================================================================

@app.route('/api/video-text/links', methods=['POST'])
@require_auth
def create_video_text_link():
    """Create a link between video segment and text segment."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['video_id', 'video_segment_id', 'corpus_id', 'text_segment_id', 'link_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create link
        link = video_text_service.create_video_text_link(
            video_id=data['video_id'],
            video_segment_id=data['video_segment_id'],
            corpus_id=data['corpus_id'],
            text_segment_id=data['text_segment_id'],
            link_type=data['link_type'],
            created_by=current_user['id'],
            confidence_score=data.get('confidence_score', 0.0),
            metadata=data.get('metadata', {}),
            notes=data.get('notes', '')
        )
        
        return jsonify({
            'message': 'Video-text link created successfully',
            'link': {
                'id': link.id,
                'video_id': link.video_id,
                'corpus_id': link.corpus_id,
                'link_type': link.link_type,
                'confidence_score': link.confidence_score,
                'created_by': link.created_by,
                'created_at': link.created_at
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating video-text link: {e}")
        return jsonify({'error': 'Failed to create video-text link'}), 500

@app.route('/api/video-text/links', methods=['GET'])
@require_auth
def list_video_text_links():
    """List video-text links with optional filtering."""
    try:
        current_user = get_current_user()
        
        # Get query parameters
        video_id = request.args.get('video_id')
        corpus_id = request.args.get('corpus_id')
        link_type = request.args.get('link_type')
        
        # Get links
        links = video_text_service.get_video_text_links(
            video_id=video_id,
            corpus_id=corpus_id,
            link_type=link_type
        )
        
        # Convert to JSON-serializable format
        links_data = []
        for link in links:
            links_data.append({
                'id': link.id,
                'video_id': link.video_id,
                'video_segment_id': link.video_segment_id,
                'corpus_id': link.corpus_id,
                'text_segment_id': link.text_segment_id,
                'link_type': link.link_type,
                'confidence_score': link.confidence_score,
                'status': link.status,
                'created_by': link.created_by,
                'created_at': link.created_at,
                'updated_at': link.updated_at
            })
        
        return jsonify({
            'links': links_data,
            'total': len(links_data)
        })
        
    except Exception as e:
        logger.error(f"Error listing video-text links: {e}")
        return jsonify({'error': 'Failed to list video-text links'}), 500

@app.route('/api/video-text/links/<link_id>', methods=['GET'])
@require_auth
def get_video_text_link(link_id):
    """Get a specific video-text link by ID."""
    try:
        current_user = get_current_user()
        
        link = video_text_service.get_video_text_link(link_id)
        if not link:
            return jsonify({'error': 'Video-text link not found'}), 404
        
        return jsonify({
            'link': {
                'id': link.id,
                'video_id': link.video_id,
                'video_segment_id': link.video_segment_id,
                'corpus_id': link.corpus_id,
                'text_segment_id': link.text_segment_id,
                'link_type': link.link_type,
                'confidence_score': link.confidence_score,
                'status': link.status,
                'metadata': link.metadata,
                'notes': link.notes,
                'created_by': link.created_by,
                'created_at': link.created_at,
                'updated_at': link.updated_at
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting video-text link {link_id}: {e}")
        return jsonify({'error': 'Failed to get video-text link'}), 500

@app.route('/api/video-text/links/<link_id>', methods=['PUT'])
@require_auth
def update_video_text_link(link_id):
    """Update a video-text link."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Update link
        updated_link = video_text_service.update_video_text_link(
            link_id=link_id,
            updates=data,
            updated_by=current_user['id']
        )
        
        if not updated_link:
            return jsonify({'error': 'Video-text link not found'}), 404
        
        return jsonify({
            'message': 'Video-text link updated successfully',
            'link': {
                'id': updated_link.id,
                'status': updated_link.status,
                'updated_at': updated_link.updated_at
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating video-text link {link_id}: {e}")
        return jsonify({'error': 'Failed to update video-text link'}), 500

@app.route('/api/video-text/links/<link_id>', methods=['DELETE'])
@require_auth
def delete_video_text_link(link_id):
    """Delete a video-text link (soft delete)."""
    try:
        current_user = get_current_user()
        
        success = video_text_service.delete_video_text_link(
            link_id=link_id,
            deleted_by=current_user['id']
        )
        
        if not success:
            return jsonify({'error': 'Video-text link not found'}), 404
        
        return jsonify({'message': 'Video-text link deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting video-text link {link_id}: {e}")
        return jsonify({'error': 'Failed to delete video-text link'}), 500

# ============================================================================
# VIDEO-TEXT ANNOTATION ENDPOINTS
# ============================================================================

@app.route('/api/video-text/annotations', methods=['POST'])
@require_auth
def create_video_text_annotation():
    """Create a video-text annotation combining video and text data."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['video_id', 'start_time', 'end_time', 'annotation_type', 'text_content']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create annotation
        annotation = video_text_service.create_video_text_annotation(
            video_id=data['video_id'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            annotation_type=data['annotation_type'],
            text_content=data['text_content'],
            created_by=current_user['id'],
            linked_corpus_id=data.get('linked_corpus_id'),
            linked_text_segment_id=data.get('linked_text_segment_id'),
            confidence_score=data.get('confidence_score', 0.0),
            hand_features=data.get('hand_features', {}),
            spatial_features=data.get('spatial_features', {}),
            temporal_features=data.get('temporal_features', {}),
            tags=data.get('tags', [])
        )
        
        return jsonify({
            'message': 'Video-text annotation created successfully',
            'annotation': {
                'id': annotation.id,
                'video_id': annotation.video_id,
                'start_time': annotation.start_time,
                'end_time': annotation.end_time,
                'annotation_type': annotation.annotation_type,
                'text_content': annotation.text_content,
                'linked_corpus_id': annotation.linked_corpus_id,
                'linked_text_segment_id': annotation.linked_text_segment_id,
                'confidence_score': annotation.confidence_score,
                'created_by': annotation.created_by,
                'created_at': annotation.created_at
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating video-text annotation: {e}")
        return jsonify({'error': 'Failed to create video-text annotation'}), 500

@app.route('/api/video-text/annotations', methods=['GET'])
@require_auth
def list_video_text_annotations():
    """List video-text annotations with optional filtering."""
    try:
        current_user = get_current_user()
        
        # Get query parameters
        video_id = request.args.get('video_id')
        annotation_type = request.args.get('annotation_type')
        status = request.args.get('status')
        
        # Get annotations
        annotations = video_text_service.get_video_text_annotations(
            video_id=video_id,
            annotation_type=annotation_type,
            status=status
        )
        
        # Convert to JSON-serializable format
        annotations_data = []
        for annotation in annotations:
            annotations_data.append({
                'id': annotation.id,
                'video_id': annotation.video_id,
                'start_time': annotation.start_time,
                'end_time': annotation.end_time,
                'duration': annotation.duration,
                'annotation_type': annotation.annotation_type,
                'text_content': annotation.text_content,
                'linked_corpus_id': annotation.linked_corpus_id,
                'linked_text_segment_id': annotation.linked_text_segment_id,
                'confidence_score': annotation.confidence_score,
                'status': annotation.status,
                'tags': annotation.tags,
                'created_by': annotation.created_by,
                'created_at': annotation.created_at,
                'updated_at': annotation.updated_at
            })
        
        return jsonify({
            'annotations': annotations_data,
            'total': len(annotations_data)
        })
        
    except Exception as e:
        logger.error(f"Error listing video-text annotations: {e}")
        return jsonify({'error': 'Failed to list video-text annotations'}), 500

# ============================================================================
# UNIFIED SEARCH ENDPOINTS
# ============================================================================

@app.route('/api/video-text/search', methods=['GET'])
@require_auth
def unified_search():
    """Search across both video and text content."""
    try:
        current_user = get_current_user()
        
        query = request.args.get('q')
        search_type = request.args.get('type', 'combined')
        video_id = request.args.get('video_id')
        corpus_id = request.args.get('corpus_id')
        annotation_type = request.args.get('annotation_type')
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        # Perform unified search
        results = video_text_service.unified_search(
            query=query,
            search_type=search_type,
            video_id=video_id,
            corpus_id=corpus_id,
            annotation_type=annotation_type
        )
        
        # Convert to JSON-serializable format
        results_data = []
        for result in results:
            results_data.append({
                'result_type': result.result_type,
                'video_id': result.video_id,
                'text_corpus_id': result.text_corpus_id,
                'text_segment_id': result.text_segment_id,
                'video_segment_id': result.video_segment_id,
                'content': result.content,
                'confidence_score': result.confidence_score,
                'metadata': result.metadata,
                'timestamp': result.timestamp,
                'relevance_score': result.relevance_score
            })
        
        return jsonify({
            'query': query,
            'search_type': search_type,
            'results': results_data,
            'total': len(results_data)
        })
        
    except Exception as e:
        logger.error(f"Error in unified search: {e}")
        return jsonify({'error': 'Failed to perform unified search'}), 500

# ============================================================================
# EXPORT ENDPOINTS
# ============================================================================

@app.route('/api/video-text/exports', methods=['POST'])
@require_auth
def create_video_text_export():
    """Create an export job for combined video-text data."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        export_format = data.get('format', 'json')
        if export_format not in ['json', 'csv', 'txt', 'combined']:
            return jsonify({'error': 'Invalid export format. Use: json, csv, txt, or combined'}), 400
        
        # Create export
        export_job = video_text_service.create_video_text_export(
            video_id=data['video_id'],
            export_format=export_format,
            created_by=current_user['id'],
            corpus_id=data.get('corpus_id'),
            export_metadata=data.get('export_metadata', {})
        )
        
        return jsonify({
            'message': 'Video-text export job created successfully',
            'export': {
                'id': export_job.id,
                'video_id': export_job.video_id,
                'corpus_id': export_job.corpus_id,
                'format': export_job.export_format,
                'status': export_job.status,
                'created_at': export_job.created_at
            }
        }), 202
        
    except Exception as e:
        logger.error(f"Error creating video-text export: {e}")
        return jsonify({'error': 'Failed to create video-text export'}), 500

@app.route('/api/video-text/exports/<export_id>', methods=['GET'])
@require_auth
def get_video_text_export_status(export_id):
    """Get the status of a video-text export job."""
    try:
        current_user = get_current_user()
        
        export_job = video_text_service.get_video_text_export_status(export_id)
        if not export_job:
            return jsonify({'error': 'Export job not found'}), 404
        
        return jsonify({
            'export': {
                'id': export_job.id,
                'video_id': export_job.video_id,
                'corpus_id': export_job.corpus_id,
                'status': export_job.status,
                'format': export_job.export_format,
                'created_at': export_job.created_at,
                'completed_at': export_job.completed_at,
                'download_url': export_job.download_url,
                'error_message': export_job.error_message
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting export status {export_id}: {e}")
        return jsonify({'error': 'Failed to get export status'}), 500

@app.route('/api/video-text/exports/<export_id>/download', methods=['GET'])
@require_auth
def download_video_text_export(export_id):
    """Download a completed video-text export."""
    try:
        current_user = get_current_user()
        
        export_job = video_text_service.get_video_text_export_status(export_id)
        if not export_job:
            return jsonify({'error': 'Export job not found'}), 404
        
        if export_job.status != 'completed':
            return jsonify({'error': 'Export job not completed yet'}), 400
        
        # For now, return a simple message
        # In production, this would generate and return the actual file
        return jsonify({
            'message': 'Video-text export ready for download',
            'export_id': export_id,
            'format': export_job.export_format,
            'download_url': export_job.download_url
        })
        
    except Exception as e:
        logger.error(f"Error downloading export {export_id}: {e}")
        return jsonify({'error': 'Failed to download export'}), 500

# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@app.route('/api/video-text/stats', methods=['GET'])
@require_auth
def get_video_text_statistics():
    """Get statistics for video-text integration."""
    try:
        current_user = get_current_user()
        
        # Get query parameters
        video_id = request.args.get('video_id')
        corpus_id = request.args.get('corpus_id')
        
        # Get statistics
        stats = video_text_service.get_video_text_statistics(
            video_id=video_id,
            corpus_id=corpus_id
        )
        
        return jsonify({
            'video_id': video_id,
            'corpus_id': corpus_id,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting video-text statistics: {e}")
        return jsonify({'error': 'Failed to get video-text statistics'}), 500

# ============================================================================
# INTEGRATION ENDPOINTS
# ============================================================================

@app.route('/api/video-text/integration/status', methods=['GET'])
@require_auth
def get_integration_status():
    """Get the status of Epic 3 integration with Epic 1 and Epic 2."""
    try:
        current_user = get_current_user()
        
        # Check Epic 1 (Authentication) status
        epic1_status = "operational"  # Assuming Epic 1 is working
        
        # Check Epic 2 (Text Corpora) status
        try:
            corpora = text_corpus_service.list_corpora()
            epic2_status = "operational"
            epic2_corpora_count = len(corpora)
        except Exception:
            epic2_status = "degraded"
            epic2_corpora_count = 0
        
        # Check Epic 3 (Video-Text) status
        try:
            annotations = video_text_service.get_video_text_annotations()
            epic3_status = "operational"
            epic3_annotations_count = len(annotations)
        except Exception:
            epic3_status = "degraded"
            epic3_annotations_count = 0
        
        return jsonify({
            'integration_status': {
                'epic1_authentication': {
                    'status': epic1_status,
                    'description': 'User authentication and role-based access control'
                },
                'epic2_text_corpora': {
                    'status': epic2_status,
                    'description': 'Text corpus management and organization',
                    'corpora_count': epic2_corpora_count
                },
                'epic3_video_text': {
                    'status': epic3_status,
                    'description': 'Video-text linking and unified workspace',
                    'annotations_count': epic3_annotations_count
                }
            },
            'overall_status': 'operational' if all(s == 'operational' for s in [epic1_status, epic2_status, epic3_status]) else 'degraded',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        return jsonify({'error': 'Failed to get integration status'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=True)
