"""
Text Corpus Management API for SPOKHAND SIGNCUT

This API provides RESTful endpoints for text corpus management, integrating with
Epic 1 authentication system and providing role-based access control.
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import os

# Import our services
from text_corpus_service import TextCorpusService
from auth_service import AuthService, require_auth, get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
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
        'service': 'text-corpus-api',
        'timestamp': datetime.utcnow().isoformat()
    })

# ============================================================================
# CORPUS MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/corpora', methods=['POST'])
@require_auth
def create_corpus():
    """Create a new text corpus."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'description', 'language']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create corpus
        corpus = text_corpus_service.create_corpus(
            name=data['name'],
            description=data['description'],
            language=data['language'],
            created_by=current_user['id'],
            metadata=data.get('metadata', {}),
            tags=data.get('tags', [])
        )
        
        return jsonify({
            'message': 'Corpus created successfully',
            'corpus': {
                'id': corpus.id,
                'name': corpus.name,
                'description': corpus.description,
                'language': corpus.language,
                'status': corpus.status,
                'created_by': corpus.created_by,
                'created_at': corpus.created_at
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating corpus: {e}")
        return jsonify({'error': 'Failed to create corpus'}), 500

@app.route('/api/corpora', methods=['GET'])
@require_auth
def list_corpora():
    """List all corpora with optional filtering."""
    try:
        current_user = get_current_user()
        
        # Get query parameters
        status = request.args.get('status')
        language = request.args.get('language')
        user_id = request.args.get('user_id')
        
        # If no user_id specified, default to current user's corpora
        if not user_id:
            user_id = current_user['id']
        
        # Check permissions - users can only see their own corpora unless they're admin/expert
        if (current_user['id'] != user_id and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        corpora = text_corpus_service.list_corpora(
            user_id=user_id,
            status=status,
            language=language
        )
        
        # Convert to JSON-serializable format
        corpora_data = []
        for corpus in corpora:
            corpora_data.append({
                'id': corpus.id,
                'name': corpus.name,
                'description': corpus.description,
                'language': corpus.language,
                'status': corpus.status,
                'total_segments': corpus.total_segments,
                'validated_segments': corpus.validated_segments,
                'tags': corpus.tags,
                'version': corpus.version,
                'created_by': corpus.created_by,
                'created_at': corpus.created_at,
                'updated_at': corpus.updated_at
            })
        
        return jsonify({
            'corpora': corpora_data,
            'total': len(corpora_data)
        })
        
    except Exception as e:
        logger.error(f"Error listing corpora: {e}")
        return jsonify({'error': 'Failed to list corpora'}), 500

@app.route('/api/corpora/<corpus_id>', methods=['GET'])
@require_auth
def get_corpus(corpus_id):
    """Get a specific corpus by ID."""
    try:
        current_user = get_current_user()
        
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        return jsonify({
            'corpus': {
                'id': corpus.id,
                'name': corpus.name,
                'description': corpus.description,
                'language': corpus.language,
                'metadata': corpus.metadata,
                'status': corpus.status,
                'total_segments': corpus.total_segments,
                'validated_segments': corpus.validated_segments,
                'tags': corpus.tags,
                'version': corpus.version,
                'created_by': corpus.created_by,
                'created_at': corpus.created_at,
                'updated_at': corpus.updated_at
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to get corpus'}), 500

@app.route('/api/corpora/<corpus_id>', methods=['PUT'])
@require_auth
def update_corpus(corpus_id):
    """Update a corpus."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Check if corpus exists
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions - only creator, admin, or expert can update
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Update corpus
        updated_corpus = text_corpus_service.update_corpus(
            corpus_id=corpus_id,
            updates=data,
            updated_by=current_user['id']
        )
        
        if not updated_corpus:
            return jsonify({'error': 'Failed to update corpus'}), 500
        
        return jsonify({
            'message': 'Corpus updated successfully',
            'corpus': {
                'id': updated_corpus.id,
                'name': updated_corpus.name,
                'description': updated_corpus.description,
                'language': updated_corpus.language,
                'status': updated_corpus.status,
                'updated_at': updated_corpus.updated_at
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to update corpus'}), 500

@app.route('/api/corpora/<corpus_id>', methods=['DELETE'])
@require_auth
def delete_corpus(corpus_id):
    """Delete a corpus (soft delete)."""
    try:
        current_user = get_current_user()
        
        # Check if corpus exists
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions - only creator or admin can delete
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Delete corpus
        success = text_corpus_service.delete_corpus(
            corpus_id=corpus_id,
            deleted_by=current_user['id']
        )
        
        if not success:
            return jsonify({'error': 'Failed to delete corpus'}), 500
        
        return jsonify({'message': 'Corpus deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to delete corpus'}), 500

# ============================================================================
# TEXT SEGMENT ENDPOINTS
# ============================================================================

@app.route('/api/corpora/<corpus_id>/segments', methods=['POST'])
@require_auth
def add_text_segment(corpus_id):
    """Add a new text segment to a corpus."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['text', 'segment_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if corpus exists
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions - only creator, admin, expert, or qualifier can add segments
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles'] and
            'qualifier' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Add segment
        segment = text_corpus_service.add_text_segment(
            corpus_id=corpus_id,
            text=data['text'],
            segment_type=data['segment_type'],
            created_by=current_user['id'],
            metadata=data.get('metadata', {}),
            related_signs=data.get('related_signs', [])
        )
        
        return jsonify({
            'message': 'Text segment added successfully',
            'segment': {
                'id': segment.id,
                'text': segment.text,
                'segment_type': segment.segment_type,
                'position': segment.position,
                'status': segment.status,
                'created_by': segment.created_by,
                'created_at': segment.created_at
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Error adding text segment to corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to add text segment'}), 500

@app.route('/api/corpora/<corpus_id>/segments', methods=['GET'])
@require_auth
def list_segments(corpus_id):
    """List all segments in a corpus."""
    try:
        current_user = get_current_user()
        
        # Check if corpus exists
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get query parameters
        status = request.args.get('status')
        
        segments = text_corpus_service.list_segments(
            corpus_id=corpus_id,
            status=status
        )
        
        # Convert to JSON-serializable format
        segments_data = []
        for segment in segments:
            segments_data.append({
                'id': segment.id,
                'text': segment.text,
                'segment_type': segment.segment_type,
                'position': segment.position,
                'status': segment.status,
                'metadata': segment.metadata,
                'validation_notes': segment.validation_notes,
                'related_signs': segment.related_signs,
                'confidence_score': segment.confidence_score,
                'created_by': segment.created_by,
                'created_at': segment.created_at,
                'updated_at': segment.updated_at
            })
        
        return jsonify({
            'segments': segments_data,
            'total': len(segments_data)
        })
        
    except Exception as e:
        logger.error(f"Error listing segments for corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to list segments'}), 500

@app.route('/api/segments/<segment_id>', methods=['GET'])
@require_auth
def get_segment(segment_id):
    """Get a specific text segment by ID."""
    try:
        current_user = get_current_user()
        
        segment = text_corpus_service.get_segment(segment_id)
        if not segment:
            return jsonify({'error': 'Segment not found'}), 404
        
        # Check permissions by getting the corpus
        corpus = text_corpus_service.get_corpus(segment.corpus_id)
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        return jsonify({
            'segment': {
                'id': segment.id,
                'corpus_id': segment.corpus_id,
                'text': segment.text,
                'segment_type': segment.segment_type,
                'position': segment.position,
                'status': segment.status,
                'metadata': segment.metadata,
                'validation_notes': segment.validation_notes,
                'related_signs': segment.related_signs,
                'confidence_score': segment.confidence_score,
                'created_by': segment.created_by,
                'created_at': segment.created_at,
                'updated_at': segment.updated_at
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting segment {segment_id}: {e}")
        return jsonify({'error': 'Failed to get segment'}), 500

@app.route('/api/segments/<segment_id>', methods=['PUT'])
@require_auth
def update_segment(segment_id):
    """Update a text segment."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        # Check if segment exists
        segment = text_corpus_service.get_segment(segment_id)
        if not segment:
            return jsonify({'error': 'Segment not found'}), 404
        
        # Check permissions by getting the corpus
        corpus = text_corpus_service.get_corpus(segment.corpus_id)
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles'] and
            'qualifier' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Update segment
        updated_segment = text_corpus_service.update_segment(
            segment_id=segment_id,
            updates=data,
            updated_by=current_user['id']
        )
        
        if not updated_segment:
            return jsonify({'error': 'Failed to update segment'}), 500
        
        return jsonify({
            'message': 'Segment updated successfully',
            'segment': {
                'id': updated_segment.id,
                'text': updated_segment.text,
                'status': updated_segment.status,
                'updated_at': updated_segment.updated_at
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating segment {segment_id}: {e}")
        return jsonify({'error': 'Failed to update segment'}), 500

@app.route('/api/segments/<segment_id>', methods=['DELETE'])
@require_auth
def delete_segment(segment_id):
    """Delete a text segment (soft delete)."""
    try:
        current_user = get_current_user()
        
        # Check if segment exists
        segment = text_corpus_service.get_segment(segment_id)
        if not segment:
            return jsonify({'error': 'Segment not found'}), 404
        
        # Check permissions by getting the corpus
        corpus = text_corpus_service.get_corpus(segment.corpus_id)
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Delete segment
        success = text_corpus_service.delete_segment(
            segment_id=segment_id,
            deleted_by=current_user['id']
        )
        
        if not success:
            return jsonify({'error': 'Failed to delete segment'}), 500
        
        return jsonify({'message': 'Segment deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting segment {segment_id}: {e}")
        return jsonify({'error': 'Failed to delete segment'}), 500

# ============================================================================
# SEARCH AND EXPORT ENDPOINTS
# ============================================================================

@app.route('/api/corpora/<corpus_id>/search', methods=['GET'])
@require_auth
def search_corpus(corpus_id):
    """Search within a corpus."""
    try:
        current_user = get_current_user()
        query = request.args.get('q')
        search_type = request.args.get('type', 'text')
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        # Check if corpus exists
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Search corpus
        results = text_corpus_service.search_corpus(
            corpus_id=corpus_id,
            query=query,
            search_type=search_type
        )
        
        # Convert to JSON-serializable format
        results_data = []
        for segment in results:
            results_data.append({
                'id': segment.id,
                'text': segment.text,
                'segment_type': segment.segment_type,
                'position': segment.position,
                'status': segment.status,
                'confidence_score': segment.confidence_score
            })
        
        return jsonify({
            'query': query,
            'search_type': search_type,
            'results': results_data,
            'total': len(results_data)
        })
        
    except Exception as e:
        logger.error(f"Error searching corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to search corpus'}), 500

@app.route('/api/corpora/<corpus_id>/export', methods=['POST'])
@require_auth
def export_corpus(corpus_id):
    """Export a corpus."""
    try:
        current_user = get_current_user()
        data = request.get_json()
        
        export_format = data.get('format', 'json')
        if export_format not in ['json', 'csv', 'txt']:
            return jsonify({'error': 'Invalid export format. Use: json, csv, or txt'}), 400
        
        # Check if corpus exists
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Create export job
        export_job = text_corpus_service.export_corpus(
            corpus_id=corpus_id,
            export_format=export_format,
            created_by=current_user['id']
        )
        
        return jsonify({
            'message': 'Export job created successfully',
            'export': {
                'id': export_job.id,
                'status': export_job.status,
                'format': export_job.export_format,
                'created_at': export_job.created_at
            }
        }), 202
        
    except Exception as e:
        logger.error(f"Error creating export for corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to create export'}), 500

@app.route('/api/corpora/exports/<export_id>', methods=['GET'])
@require_auth
def get_export_status(export_id):
    """Get the status of an export job."""
    try:
        current_user = get_current_user()
        
        export_job = text_corpus_service.get_export_status(export_id)
        if not export_job:
            return jsonify({'error': 'Export job not found'}), 404
        
        # Check permissions by getting the corpus
        corpus = text_corpus_service.get_corpus(export_job.corpus_id)
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        return jsonify({
            'export': {
                'id': export_job.id,
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

@app.route('/api/corpora/exports/<export_id>/download', methods=['GET'])
@require_auth
def download_export(export_id):
    """Download a completed export."""
    try:
        current_user = get_current_user()
        
        export_job = text_corpus_service.get_export_status(export_id)
        if not export_job:
            return jsonify({'error': 'Export job not found'}), 404
        
        if export_job.status != 'completed':
            return jsonify({'error': 'Export job not completed yet'}), 400
        
        # Check permissions by getting the corpus
        corpus = text_corpus_service.get_corpus(export_job.corpus_id)
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # For now, return a simple message
        # In production, this would generate and return the actual file
        return jsonify({
            'message': 'Export ready for download',
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

@app.route('/api/corpora/<corpus_id>/stats', methods=['GET'])
@require_auth
def get_corpus_stats(corpus_id):
    """Get statistics for a corpus."""
    try:
        current_user = get_current_user()
        
        # Check if corpus exists
        corpus = text_corpus_service.get_corpus(corpus_id)
        if not corpus:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Check permissions
        if (corpus.created_by != current_user['id'] and 
            'admin' not in current_user['roles'] and 
            'expert' not in current_user['roles']):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get segments for statistics
        segments = text_corpus_service.list_segments(corpus_id)
        
        # Calculate statistics
        total_segments = len(segments)
        draft_segments = len([s for s in segments if s.status == 'draft'])
        validated_segments = len([s for s in segments if s.status == 'validated'])
        approved_segments = len([s for s in segments if s.status == 'approved'])
        rejected_segments = len([s for s in segments if s.status == 'rejected'])
        
        # Segment type distribution
        segment_types = {}
        for segment in segments:
            seg_type = segment.segment_type
            segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
        
        return jsonify({
            'corpus_id': corpus_id,
            'total_segments': total_segments,
            'status_distribution': {
                'draft': draft_segments,
                'validated': validated_segments,
                'approved': approved_segments,
                'rejected': rejected_segments
            },
            'segment_type_distribution': segment_types,
            'validation_rate': (validated_segments / total_segments * 100) if total_segments > 0 else 0,
            'approval_rate': (approved_segments / total_segments * 100) if total_segments > 0 else 0
        })
        
    except Exception as e:
        logger.error(f"Error getting stats for corpus {corpus_id}: {e}")
        return jsonify({'error': 'Failed to get corpus statistics'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)
