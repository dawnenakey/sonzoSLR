#!/usr/bin/env python3
"""
Script to upload WLASL videos from local directory to AWS S3
"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import boto3
from datetime import datetime

# Add the src directory to the path so we can import our handlers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from aws.wlasl_upload_handler import WLASLUploadHandler

def find_wlasl_videos(directory: str, extensions: List[str] = None) -> List[Dict]:
    """
    Find all WLASL video files in the given directory and subdirectories
    
    Args:
        directory: Root directory to search
        extensions: List of video file extensions to look for
        
    Returns:
        List of dictionaries containing file paths and metadata
    """
    if extensions is None:
        extensions = ['.webm', '.mp4', '.avi', '.mov', '.mkv']
    
    video_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Directory {directory} does not exist!")
        return video_files
    
    print(f"Scanning directory: {directory}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            file_ext = file_path.suffix.lower()
            
            if file_ext in extensions:
                # Try to extract metadata from filename or path
                metadata = extract_metadata_from_path(file_path)
                
                video_files.append({
                    'file_path': str(file_path),
                    'metadata': metadata
                })
    
    print(f"Found {len(video_files)} video files")
    return video_files

def extract_metadata_from_path(file_path: Path) -> Dict:
    """
    Extract metadata from file path and name
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        'filename': file_path.name,
        'file_size': file_path.stat().st_size,
        'created_at': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
    }
    
    # Try to extract gloss (sign name) from filename
    filename = file_path.stem.lower()
    
    # Common patterns in WLASL filenames
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 2:
            # Assume first part might be gloss
            metadata['gloss'] = parts[0]
            # Check if there's a split indicator
            if any(part in ['train', 'val', 'test'] for part in parts):
                for part in parts:
                    if part in ['train', 'val', 'test']:
                        metadata['split'] = part
                        break
            else:
                metadata['split'] = 'unknown'
    else:
        metadata['gloss'] = filename
        metadata['split'] = 'unknown'
    
    # Try to extract instance ID from filename
    if any(char.isdigit() for char in filename):
        # Extract numbers that might be instance ID
        import re
        numbers = re.findall(r'\d+', filename)
        if numbers:
            metadata['instance_id'] = numbers[0]
    
    return metadata

def load_wlasl_json(json_file: str) -> Dict:
    """
    Load WLASL JSON file to get official metadata
    
    Args:
        json_file: Path to WLASL_v0.3.json file
        
    Returns:
        Dictionary mapping gloss to metadata
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create a mapping from gloss to metadata
        gloss_metadata = {}
        for entry in data:
            gloss = entry.get('gloss', '').lower()
            gloss_metadata[gloss] = entry
        
        return gloss_metadata
    except Exception as e:
        print(f"Warning: Could not load WLASL JSON file: {e}")
        return {}

def match_video_with_metadata(video_files: List[Dict], wlasl_metadata: Dict) -> List[Dict]:
    """
    Match video files with official WLASL metadata
    
    Args:
        video_files: List of video file dictionaries
        wlasl_metadata: Dictionary of WLASL metadata
        
    Returns:
        List of video files with enhanced metadata
    """
    enhanced_files = []
    
    for video in video_files:
        metadata = video['metadata']
        gloss = metadata.get('gloss', '').lower()
        
        # Try to match with official WLASL metadata
        if gloss in wlasl_metadata:
            official_metadata = wlasl_metadata[gloss]
            
            # Enhance metadata with official data
            metadata.update({
                'official_gloss': official_metadata.get('gloss', gloss),
                'instances_count': len(official_metadata.get('instances', [])),
                'category': official_metadata.get('category', ''),
                'official_metadata': official_metadata
            })
        
        enhanced_files.append({
            'file_path': video['file_path'],
            'metadata': metadata
        })
    
    return enhanced_files

def upload_videos_to_aws(video_files: List[Dict], batch_size: int = 10) -> List[Dict]:
    """
    Upload videos to AWS S3 using the WLASLUploadHandler
    
    Args:
        video_files: List of video file dictionaries
        batch_size: Number of videos to upload in each batch
        
    Returns:
        List of upload results
    """
    handler = WLASLUploadHandler()
    results = []
    
    print(f"Starting upload of {len(video_files)} videos...")
    
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i + batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(len(video_files) + batch_size - 1)//batch_size}")
        
        batch_results = handler.batch_upload_wlasl_videos(batch)
        results.extend(batch_results)
        
        # Print progress
        successful = sum(1 for r in batch_results if r.get('success', False))
        print(f"  Successfully uploaded {successful}/{len(batch)} videos in this batch")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Upload WLASL videos to AWS S3')
    parser.add_argument('directory', help='Directory containing WLASL videos')
    parser.add_argument('--wlasl-json', help='Path to WLASL_v0.3.json file for metadata')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of videos to upload in each batch')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without actually uploading')
    parser.add_argument('--extensions', nargs='+', default=['.webm', '.mp4'], help='Video file extensions to look for')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist!")
        sys.exit(1)
    
    # Find video files
    print("Scanning for video files...")
    video_files = find_wlasl_videos(args.directory, args.extensions)
    
    if not video_files:
        print("No video files found!")
        sys.exit(0)
    
    # Load WLASL metadata if provided
    wlasl_metadata = {}
    if args.wlasl_json and os.path.exists(args.wlasl_json):
        print("Loading WLASL metadata...")
        wlasl_metadata = load_wlasl_json(args.wlasl_json)
        print(f"Loaded metadata for {len(wlasl_metadata)} glosses")
    
    # Match videos with metadata
    enhanced_files = match_video_with_metadata(video_files, wlasl_metadata)
    
    # Show summary
    print(f"\nFound {len(enhanced_files)} video files:")
    for i, video in enumerate(enhanced_files[:5]):  # Show first 5
        metadata = video['metadata']
        print(f"  {i+1}. {metadata['filename']} (gloss: {metadata.get('gloss', 'unknown')})")
    
    if len(enhanced_files) > 5:
        print(f"  ... and {len(enhanced_files) - 5} more files")
    
    if args.dry_run:
        print("\nDRY RUN - No files will be uploaded")
        return
    
    # Confirm upload
    response = input(f"\nUpload {len(enhanced_files)} videos to AWS? (y/N): ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        return
    
    # Upload to AWS
    print("\nStarting upload to AWS...")
    results = upload_videos_to_aws(enhanced_files, args.batch_size)
    
    # Show results
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"\nUpload complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed uploads:")
        for result in results:
            if not result.get('success', False):
                print(f"  - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 