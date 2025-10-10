#!/usr/bin/env python3
"""
ASLLRP Integration Setup Script

This script helps you set up the ASLLRP integration for sentence-level ASL recognition.
Run this after downloading SignStream® and getting ASLLRP data access.

Usage:
    python setup_asllrp_integration.py --data_path /path/to/asllrp/data
"""

import argparse
import os
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_asllrp_integration(data_path: str):
    """Set up ASLLRP integration with the provided data path."""
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please ensure you have downloaded ASLLRP data to this location")
        return False
    
    logger.info(f"Setting up ASLLRP integration with data path: {data_path}")
    
    # Check for required files
    required_extensions = ['.mp4', '.avi', '.mov', '.xml', '.json', '.csv']
    found_files = {}
    
    for ext in required_extensions:
        files = list(data_path.rglob(f'*{ext}'))
        found_files[ext] = len(files)
        logger.info(f"Found {len(files)} {ext} files")
    
    # Create output directories
    output_dirs = ['processed_data', 'training_data', 'models', 'logs']
    for dir_name in output_dirs:
        output_dir = data_path.parent / dir_name
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create configuration file
    config = {
        'asllrp_data_path': str(data_path),
        'processed_data_path': str(data_path.parent / 'processed_data'),
        'training_data_path': str(data_path.parent / 'training_data'),
        'models_path': str(data_path.parent / 'models'),
        'logs_path': str(data_path.parent / 'logs'),
        'vocabulary_size': 0,  # Will be updated after processing
        'max_sequence_length': 50,
        'model_config': {
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1
        }
    }
    
    config_path = data_path.parent / 'asllrp_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created configuration file: {config_path}")
    
    # Test data loading
    try:
        from asllrp_integration import ASLLRPDataLoader
        
        logger.info("Testing ASLLRP data loading...")
        loader = ASLLRPDataLoader(str(data_path))
        videos = loader.load_dataset()
        
        if videos:
            stats = loader.get_sentence_statistics()
            logger.info("Data loading successful!")
            logger.info(f"Dataset statistics: {stats}")
            
            # Update config with actual vocabulary size
            config['vocabulary_size'] = stats.get('vocabulary_size', 0)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        else:
            logger.warning("No videos loaded. Check data format and structure.")
            
    except ImportError as e:
        logger.error(f"Could not import ASLLRP integration module: {e}")
        logger.info("Make sure src/asllrp_integration.py is in your Python path")
        return False
    except Exception as e:
        logger.error(f"Error testing data loading: {e}")
        return False
    
    logger.info("ASLLRP integration setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Review the configuration file: asllrp_config.json")
    logger.info("2. Run: python src/train_sentence_asllrp.py")
    logger.info("3. Check logs for training progress")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup ASLLRP integration for sentence-level ASL recognition')
    parser.add_argument('--data_path', required=True, help='Path to ASLLRP data directory')
    parser.add_argument('--test_only', action='store_true', help='Only test data loading, do not create directories')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Just test data loading
        try:
            from asllrp_integration import ASLLRPDataLoader
            loader = ASLLRPDataLoader(args.data_path)
            videos = loader.load_dataset()
            stats = loader.get_sentence_statistics()
            print(f"Test successful! Found {len(videos)} videos")
            print(f"Statistics: {stats}")
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        # Full setup
        setup_asllrp_integration(args.data_path)

if __name__ == "__main__":
    main()
