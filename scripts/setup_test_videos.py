#!/usr/bin/env python3
"""
Script to create a test directory with sample video files for testing upload functionality
"""

import os
import shutil
from pathlib import Path

def create_test_video_directory():
    """Create a test directory with sample video files"""
    
    # Create test directory structure
    test_dir = Path("test_wlasl_videos")
    test_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (test_dir / "train").mkdir(exist_ok=True)
    (test_dir / "val").mkdir(exist_ok=True)
    (test_dir / "test").mkdir(exist_ok=True)
    
    # Create sample video files (empty files for testing)
    sample_videos = [
        ("train", "hello_001.webm"),
        ("train", "goodbye_002.webm"),
        ("train", "thank_you_003.webm"),
        ("val", "hello_004.webm"),
        ("val", "goodbye_005.webm"),
        ("test", "hello_006.webm"),
        ("test", "thank_you_007.webm"),
    ]
    
    for split, filename in sample_videos:
        file_path = test_dir / split / filename
        # Create an empty file (you can replace with actual video files)
        file_path.touch()
        print(f"Created: {file_path}")
    
    print(f"\nTest directory created at: {test_dir.absolute()}")
    print("You can now test the upload script with:")
    print(f"python scripts/upload_wlasl_videos.py {test_dir.absolute()} --dry-run")
    
    return test_dir

def download_wlasl_info():
    """Provide information about downloading WLASL dataset"""
    
    print("\n" + "="*60)
    print("WLASL DATASET DOWNLOAD INFORMATION")
    print("="*60)
    print("""
To get actual WLASL videos, you need to:

1. Visit the WLASL dataset website:
   https://github.com/dxli94/WLASL

2. Download the dataset:
   - Request access to the dataset
   - Download the video files
   - Extract them to a directory

3. Organize your videos in this structure:
   wlasl_videos/
   ├── train/
   │   ├── hello_001.webm
   │   ├── goodbye_002.webm
   │   └── ...
   ├── val/
   │   ├── hello_003.webm
   │   └── ...
   └── test/
       ├── hello_004.webm
       └── ...

4. Use the upload script:
   python scripts/upload_wlasl_videos.py /path/to/your/wlasl_videos
""")

def main():
    print("WLASL Video Upload Test Setup")
    print("="*40)
    
    # Create test directory
    test_dir = create_test_video_directory()
    
    # Show download information
    download_wlasl_info()
    
    print(f"\n✅ Test setup complete!")
    print(f"📁 Test directory: {test_dir.absolute()}")
    print(f"🧪 You can test with: python scripts/upload_wlasl_videos.py {test_dir.absolute()} --dry-run")

if __name__ == "__main__":
    main() 