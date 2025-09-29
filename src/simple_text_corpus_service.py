"""
Simple Text Corpus Service for SPOKHAND SIGNCUT
Handles basic text corpus management for local testing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Simple in-memory storage for testing
corpora = {}
segments = {}
exports = {}

@app.route('/api/text-corpus/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'text_corpus',
        'timestamp': datetime.utcnow().isoformat(),
        'corpora_count': len(corpora),
        'segments_count': len(segments)
    }), 200

@app.route('/api/text-corpus/upload', methods=['POST'])
def upload_text():
    """Upload text content"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        corpus_id = data.get('corpus_id', str(uuid.uuid4()))
        
        if not text:
            return jsonify({'error': 'Text content is required'}), 400
        
        # Create corpus if it doesn't exist
        if corpus_id not in corpora:
            corpora[corpus_id] = {
                'id': corpus_id,
                'name': data.get('name', 'Untitled Corpus'),
                'description': data.get('description', ''),
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active'
            }
        
        # Add text segment
        segment_id = str(uuid.uuid4())
        segments[segment_id] = {
            'id': segment_id,
            'corpus_id': corpus_id,
            'text': text,
            'segment_type': data.get('segment_type', 'text'),
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        return jsonify({
            'message': 'Text uploaded successfully',
            'corpus_id': corpus_id,
            'segment_id': segment_id
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-corpus/process', methods=['POST'])
def process_text():
    """Process text content"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text content is required'}), 400
        
        # Simple text processing
        processed_text = {
            'original': text,
            'word_count': len(text.split()),
            'character_count': len(text),
            'processed_at': datetime.utcnow().isoformat(),
            'status': 'processed'
        }
        
        return jsonify({
            'message': 'Text processed successfully',
            'result': processed_text
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-corpus/export', methods=['POST'])
def export_corpus():
    """Export corpus data"""
    try:
        data = request.get_json()
        corpus_id = data.get('corpus_id')
        export_format = data.get('format', 'json')
        
        if not corpus_id:
            return jsonify({'error': 'Corpus ID is required'}), 400
        
        if corpus_id not in corpora:
            return jsonify({'error': 'Corpus not found'}), 404
        
        # Create export
        export_id = str(uuid.uuid4())
        exports[export_id] = {
            'id': export_id,
            'corpus_id': corpus_id,
            'format': export_format,
            'status': 'completed',
            'created_at': datetime.utcnow().isoformat(),
            'download_url': f'/api/text-corpus/download/{export_id}'
        }
        
        return jsonify({
            'message': 'Export created successfully',
            'export_id': export_id,
            'download_url': f'/api/text-corpus/download/{export_id}'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-corpus/exports', methods=['GET'])
def list_exports():
    """List all exports"""
    try:
        export_list = list(exports.values())
        return jsonify({
            'exports': export_list,
            'count': len(export_list)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-corpus/download/<export_id>', methods=['GET'])
def download_export(export_id):
    """Download export data"""
    try:
        if export_id not in exports:
            return jsonify({'error': 'Export not found'}), 404
        
        export = exports[export_id]
        
        # Get corpus data
        corpus = corpora.get(export['corpus_id'], {})
        corpus_segments = [s for s in segments.values() if s['corpus_id'] == export['corpus_id']]
        
        export_data = {
            'corpus': corpus,
            'segments': corpus_segments,
            'export_info': export
        }
        
        return jsonify(export_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Text Corpus Service on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)
