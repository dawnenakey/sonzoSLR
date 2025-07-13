#!/usr/bin/env python3
"""
Script to upload ASL-LEX CSV files from local directory to AWS S3
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add the src directory to the path so we can import our handlers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from aws.asl_lex_handler import ASLLexHandler

def find_csv_files(directory: str) -> List[Dict]:
    """
    Find all CSV files in the given directory and subdirectories
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of dictionaries containing file paths and metadata
    """
    csv_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Directory {directory} does not exist!")
        return csv_files
    
    print(f"Scanning directory: {directory}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.csv':
                # Extract metadata from the CSV file
                metadata = extract_csv_metadata(file_path)
                
                csv_files.append({
                    'file_path': str(file_path),
                    'metadata': metadata
                })
    
    print(f"Found {len(csv_files)} CSV files")
    return csv_files

def extract_csv_metadata(file_path: Path) -> Dict:
    """
    Extract metadata from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing extracted metadata
    """
    try:
        # Read CSV to get basic information
        df = pd.read_csv(file_path)
        
        metadata = {
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'description': f"ASL-LEX dataset file: {file_path.name}"
        }
        
        # Try to identify the type of ASL-LEX data based on columns
        columns_lower = [col.lower() for col in df.columns]
        
        if 'gloss' in columns_lower:
            metadata['data_type'] = 'gloss_data'
        elif 'handshape' in columns_lower:
            metadata['data_type'] = 'handshape_data'
        elif 'location' in columns_lower:
            metadata['data_type'] = 'location_data'
        elif 'movement' in columns_lower:
            metadata['data_type'] = 'movement_data'
        else:
            metadata['data_type'] = 'general_data'
        
        # Add sample data for preview
        if len(df) > 0:
            metadata['sample_data'] = df.head(3).to_dict('records')
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not read CSV file {file_path}: {e}")
        return {
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'description': f"ASL-LEX dataset file: {file_path.name}",
            'error': str(e)
        }

def upload_csv_files_to_aws(csv_files: List[Dict]) -> List[Dict]:
    """
    Upload CSV files to AWS S3 using the ASLLexHandler
    
    Args:
        csv_files: List of CSV file dictionaries
        
    Returns:
        List of upload results
    """
    handler = ASLLexHandler()
    results = []
    
    print(f"Starting upload of {len(csv_files)} CSV files...")
    
    for i, csv_file in enumerate(csv_files):
        print(f"Uploading {i+1}/{len(csv_files)}: {csv_file['metadata']['filename']}")
        
        result = handler.upload_asl_lex_csv(
            csv_file['file_path'], 
            csv_file['metadata']
        )
        results.append(result)
        
        if result.get('success', False):
            print(f"  ✓ Successfully uploaded")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Upload ASL-LEX CSV files to AWS S3')
    parser.add_argument('directory', help='Directory containing ASL-LEX CSV files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without actually uploading')
    parser.add_argument('--preview', action='store_true', help='Show preview of CSV data before uploading')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist!")
        sys.exit(1)
    
    # Find CSV files
    print("Scanning for CSV files...")
    csv_files = find_csv_files(args.directory)
    
    if not csv_files:
        print("No CSV files found!")
        sys.exit(0)
    
    # Show summary
    print(f"\nFound {len(csv_files)} CSV files:")
    for i, csv_file in enumerate(csv_files):
        metadata = csv_file['metadata']
        print(f"  {i+1}. {metadata['filename']}")
        print(f"      Rows: {metadata.get('row_count', 'unknown')}")
        print(f"      Columns: {metadata.get('column_count', 'unknown')}")
        print(f"      Type: {metadata.get('data_type', 'unknown')}")
        
        if args.preview and 'sample_data' in metadata:
            print(f"      Sample data:")
            for j, row in enumerate(metadata['sample_data'][:2]):
                print(f"        Row {j+1}: {list(row.items())[:3]}...")
        print()
    
    if args.dry_run:
        print("DRY RUN - No files will be uploaded")
        return
    
    # Confirm upload
    response = input(f"Upload {len(csv_files)} CSV files to AWS? (y/N): ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        return
    
    # Upload to AWS
    print("\nStarting upload to AWS...")
    results = upload_csv_files_to_aws(csv_files)
    
    # Show results
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"\nUpload complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if successful > 0:
        print(f"\nSuccessfully uploaded files:")
        for result in results:
            if result.get('success', False):
                metadata = result.get('metadata', {})
                print(f"  - {metadata.get('filename', 'unknown')}")
                print(f"    File ID: {result.get('file_id', 'unknown')}")
                print(f"    Rows: {metadata.get('row_count', 'unknown')}")
                print(f"    Columns: {metadata.get('column_count', 'unknown')}")
    
    if failed > 0:
        print(f"\nFailed uploads:")
        for result in results:
            if not result.get('success', False):
                print(f"  - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 