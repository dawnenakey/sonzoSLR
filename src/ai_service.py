"""
AI Service API Endpoints for Epic 4: AI Integration

This service provides REST API endpoints for the advanced sign spotting
and disambiguation features described in the research paper.
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Any
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError

# Import our sign spotting service
from sign_spotting_service import AdvancedSignSpottingService, SignSegment

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

# Initialize sign spotting service
sign_spotting_service = AdvancedSignSpottingService()

@app.route('/api/ai/analyze-video', methods=['POST'])
def analyze_video():
    """
    Analyze a video using advanced sign spotting and disambiguation.
    
    Expected payload:
    {
        "video_id": "string",
        "fusion_strategy": "late_fusion|intermediate_fusion|full_ensemble",
        "vocabulary_size": 1000,
        "beam_width": 5,
        "alpha": 0.9,
        "window_size": 16,
        "stride": 1
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['video_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        video_id = data['video_id']
        fusion_strategy = data.get('fusion_strategy', 'late_fusion')
        vocabulary_size = data.get('vocabulary_size', 1000)
        beam_width = data.get('beam_width', 5)
        alpha = data.get('alpha', 0.9)
        window_size = data.get('window_size', 16)
        stride = data.get('stride', 1)
        
        # Get video from S3
        video_url = get_video_url(video_id)
        if not video_url:
            return jsonify({'error': 'Video not found'}), 404
        
        # Download video to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_video_path = temp_file.name
            download_video_from_s3(video_id, temp_video_path)
        
        try:
            # Process video with sign spotting service
            segments = sign_spotting_service.process_video(
                temp_video_path,
                window_size=window_size,
                stride=stride
            )
            
            # Convert segments to API format
            results = {
                'video_id': video_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'fusion_strategy': fusion_strategy,
                'vocabulary_size': vocabulary_size,
                'beam_width': beam_width,
                'alpha': alpha,
                'total_segments': len(segments),
                'segments': []
            }
            
            for segment in segments:
                segment_data = {
                    'id': f"{video_id}_{segment.start_time}_{segment.end_time}",
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'sign': segment.sign,
                    'confidence': segment.confidence,
                    'hand_shape': segment.hand_shape,
                    'location': segment.location,
                    'alternatives': segment.alternatives,
                    'llm_score': segment.llm_score,
                    'features': {
                        'i3d_dimensions': len(segment.i3d_features),
                        'hand_dimensions': len(segment.hand_features)
                    }
                }
                results['segments'].append(segment_data)
            
            # Save results to DynamoDB
            save_analysis_results(video_id, results)
            
            return jsonify(results)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_video_path)
            
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/ai/analysis-results/<video_id>', methods=['GET'])
def get_analysis_results(video_id):
    """Get analysis results for a specific video."""
    try:
        results = get_analysis_results_from_db(video_id)
        if not results:
            return jsonify({'error': 'Analysis results not found'}), 404
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting analysis results: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/ai/export-results/<video_id>', methods=['GET'])
def export_analysis_results(video_id):
    """Export analysis results as JSON file."""
    try:
        results = get_analysis_results_from_db(video_id)
        if not results:
            return jsonify({'error': 'Analysis results not found'}), 404
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(results, temp_file, indent=2)
            temp_file_path = temp_file.name
        
        return send_file(
            temp_file_path,
            as_attachment=True,
            download_name=f'sign_analysis_{video_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/ai/fusion-strategies', methods=['GET'])
def get_fusion_strategies():
    """Get available fusion strategies and their descriptions."""
    strategies = [
        {
            'id': 'late_fusion',
            'name': 'Late Fusion',
            'description': 'S_Late = α * S_I3D + (1-α) * S_RH (Equation 4)',
            'performance': 'Best overall performance',
            'alpha_default': 0.9
        },
        {
            'id': 'intermediate_fusion',
            'name': 'Intermediate Fusion',
            'description': 'F_Mid = F_I3D ⊕ F_RH ⊕ F_LH ∈ R^5120',
            'performance': 'Good for complex sequences',
            'alpha_default': 0.6
        },
        {
            'id': 'full_ensemble',
            'name': 'Full Ensemble',
            'description': 'S_Ensemble = α * S_Mid + (1-α) * S_I3D (Equation 5)',
            'performance': 'Best accuracy, slower processing',
            'alpha_default': 0.6
        }
    ]
    
    return jsonify({'strategies': strategies})

@app.route('/api/ai/vocabulary-options', methods=['GET'])
def get_vocabulary_options():
    """Get available vocabulary sizes and their descriptions."""
    options = [
        {
            'size': 1000,
            'name': 'Paper Default',
            'description': '1,000 ASL signs as specified in the research paper',
            'performance': 'Fastest processing'
        },
        {
            'size': 1500,
            'name': 'Extended',
            'description': '1,500 ASL signs for broader coverage',
            'performance': 'Good balance'
        },
        {
            'size': 2000,
            'name': 'Comprehensive',
            'description': '2,000 ASL signs for comprehensive coverage',
            'performance': 'Slower but more complete'
        },
        {
            'size': 4373,
            'name': 'Maximum',
            'description': '4,373 ASL signs (maximum available)',
            'performance': 'Slowest but most complete'
        }
    ]
    
    return jsonify({'vocabulary_options': options})

@app.route('/api/ai/health', methods=['GET'])
def health_check():
    """Health check endpoint for the AI service."""
    try:
        # Check if models are loaded
        models_loaded = (
            sign_spotting_service.i3d_extractor.model is not None and
            sign_spotting_service.hand_extractor.model is not None and
            sign_spotting_service.dictionary.sign_dictionary is not None
        )
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': models_loaded,
            'fusion_strategy': sign_spotting_service.fusion_strategy,
            'vocabulary_size': sign_spotting_service.dictionary.vocabulary_size,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def get_video_url(video_id: str) -> Optional[str]:
    """Get video URL from S3."""
    try:
        # Check if video exists in S3
        s3_client.head_object(Bucket=S3_BUCKET, Key=f"videos/{video_id}")
        
        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': f"videos/{video_id}"},
            ExpiresIn=3600  # 1 hour
        )
        return url
        
    except ClientError:
        return None

def download_video_from_s3(video_id: str, local_path: str) -> bool:
    """Download video from S3 to local path."""
    try:
        s3_client.download_file(S3_BUCKET, f"videos/{video_id}", local_path)
        return True
    except ClientError as e:
        logger.error(f"Error downloading video {video_id}: {str(e)}")
        return False

def save_analysis_results(video_id: str, results: Dict[str, Any]) -> bool:
    """Save analysis results to DynamoDB."""
    try:
        table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-ai-analysis-results")
        
        item = {
            'video_id': video_id,
            'analysis_timestamp': results['analysis_timestamp'],
            'fusion_strategy': results['fusion_strategy'],
            'vocabulary_size': results['vocabulary_size'],
            'beam_width': results['beam_width'],
            'alpha': results['alpha'],
            'total_segments': results['total_segments'],
            'segments': results['segments'],
            'created_at': datetime.now().isoformat()
        }
        
        table.put_item(Item=item)
        return True
        
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        return False

def get_analysis_results_from_db(video_id: str) -> Optional[Dict[str, Any]]:
    """Get analysis results from DynamoDB."""
    try:
        table = dynamodb.Table(f"{DYNAMODB_TABLE_PREFIX}-ai-analysis-results")
        
        response = table.get_item(Key={'video_id': video_id})
        
        if 'Item' in response:
            return response['Item']
        return None
        
    except Exception as e:
        logger.error(f"Error getting analysis results: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
