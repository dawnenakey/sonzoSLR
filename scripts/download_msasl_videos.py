import json
import os
import random
import subprocess
import pandas as pd

def load_msasl_data(json_file):
    """Load MS-ASL data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    if 'youtube.com/watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    return url  # Assume it's already a video ID

def download_video(video_url, output_path, class_name):
    """Download a single video from YouTube using yt-dlp."""
    try:
        # Extract video ID
        video_id = extract_video_id(video_url)
        full_url = f'https://www.youtube.com/watch?v={video_id}'
        print(f"Attempting to download: {full_url}")
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(output_path, str(class_name))
        os.makedirs(class_dir, exist_ok=True)
        
        # Construct the output filename
        output_file = os.path.join(class_dir, f"{video_id}.mp4")
        
        # Use yt-dlp to download the video
        command = [
            'yt-dlp',
            '-f', 'best[ext=mp4]',  # Best quality MP4
            '-o', output_file,  # Output file
            full_url  # Video URL
        ]
        
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded video {video_id} for class {class_name}")
            return True
        else:
            print(f"Error downloading video {video_id}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error downloading video {video_url}: {str(e)}")
        return False

def main():
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'msasl')
    os.makedirs(data_dir, exist_ok=True)
    
    # Load MS-ASL data using absolute path
    msasl_path = os.path.expanduser('~/Downloads/MS-ASL/MSASL_train.json')
    train_data = load_msasl_data(msasl_path)
    
    # Get unique classes
    classes = list(set(item['label'] for item in train_data))
    
    # Select 5 random classes
    selected_classes = random.sample(classes, 5)
    print(f"Selected classes: {selected_classes}")
    
    # For each selected class, try to download videos
    for class_name in selected_classes:
        # Get videos for this class
        class_videos = [item for item in train_data if item['label'] == class_name]
        
        # Try up to 10 videos per class to get 2 successful downloads
        max_attempts = 10
        successful_downloads = 0
        attempts = 0
        
        print(f"\nDownloading videos for class: {class_name}")
        while successful_downloads < 2 and attempts < max_attempts:
            if attempts >= len(class_videos):
                print(f"No more videos available for class {class_name}")
                break
                
            video = class_videos[attempts]
            if download_video(video['url'], data_dir, class_name):
                successful_downloads += 1
            attempts += 1
        
        print(f"Successfully downloaded {successful_downloads} videos for class {class_name} after {attempts} attempts")

if __name__ == "__main__":
    main() 