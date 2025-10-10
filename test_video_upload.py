#!/usr/bin/env python3
"""
Test script to verify video upload and listing functionality

This script tests the fixed video upload and listing endpoints.
"""

import requests
import json
import os
from datetime import datetime

# Configuration
API_BASE_URL = "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod"

def test_session_creation():
    """Test creating a session"""
    print("🧪 Testing session creation...")
    
    response = requests.post(f"{API_BASE_URL}/sessions", json={
        "name": f"Test Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "description": "Test session for video upload"
    })
    
    if response.status_code == 201:
        data = response.json()
        print(f"✅ Session created: {data.get('session_id')}")
        return data.get('session_id')
    else:
        print(f"❌ Session creation failed: {response.status_code} - {response.text}")
        return None

def test_video_listing():
    """Test listing videos"""
    print("\n🧪 Testing video listing...")
    
    response = requests.get(f"{API_BASE_URL}/videos")
    
    if response.status_code == 200:
        data = response.json()
        videos = data.get('videos', [])
        print(f"✅ Found {len(videos)} videos")
        for video in videos[:3]:  # Show first 3 videos
            print(f"   - {video.get('filename')} ({video.get('status')})")
        return True
    else:
        print(f"❌ Video listing failed: {response.status_code} - {response.text}")
        return False

def test_upload_url_generation(session_id):
    """Test generating upload URL"""
    print(f"\n🧪 Testing upload URL generation for session {session_id}...")
    
    response = requests.post(f"{API_BASE_URL}/sessions/{session_id}/upload-video")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print(f"✅ Upload URL generated successfully")
            print(f"   Video ID: {data.get('video', {}).get('id')}")
            return True
        else:
            print(f"❌ Upload URL generation failed: {data.get('error')}")
            return False
    else:
        print(f"❌ Upload URL generation failed: {response.status_code} - {response.text}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing SpokHand Video Upload Functionality")
    print("=" * 50)
    
    # Test 1: Create session
    session_id = test_session_creation()
    if not session_id:
        print("\n❌ Cannot proceed without session")
        return
    
    # Test 2: Generate upload URL
    upload_success = test_upload_url_generation(session_id)
    
    # Test 3: List videos
    listing_success = test_video_listing()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Session Creation: {'✅ PASS' if session_id else '❌ FAIL'}")
    print(f"   Upload URL Generation: {'✅ PASS' if upload_success else '❌ FAIL'}")
    print(f"   Video Listing: {'✅ PASS' if listing_success else '❌ FAIL'}")
    
    if session_id and upload_success and listing_success:
        print("\n🎉 All tests passed! Video upload functionality is working.")
    else:
        print("\n⚠️  Some tests failed. Check the backend implementation.")

if __name__ == "__main__":
    main()
