#!/bin/bash

# Test script for complete Spokhand upload and annotation flow
API_URL="https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod"

echo "🧪 Testing Complete Spokhand Flow"
echo "=================================="

# 1. Create a session
echo ""
echo "1️⃣ Creating a test session..."
echo "-------------------------------------------"
SESSION_RESPONSE=$(curl -s -X POST "$API_URL/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Complete Flow Test Session",
    "description": "Testing complete upload and annotation flow"
  }')

echo "Session response: $SESSION_RESPONSE"

# Extract session ID from response
if command -v jq &> /dev/null; then
    SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.session_id')
else
    SESSION_ID=$(echo $SESSION_RESPONSE | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
fi

if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
    echo "❌ Failed to extract session ID from response"
    echo "Response: $SESSION_RESPONSE"
    exit 1
fi

echo "✅ Session created with ID: $SESSION_ID"

# 2. Generate upload URL
echo ""
echo "2️⃣ Generating upload URL..."
echo "-------------------------------------------"
UPLOAD_RESPONSE=$(curl -s -X POST "$API_URL/sessions/$SESSION_ID/upload-video" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test_video.mp4",
    "contentType": "video/mp4"
  }')

echo "Upload response: $UPLOAD_RESPONSE"

# Extract video ID from response
if command -v jq &> /dev/null; then
    VIDEO_ID=$(echo $UPLOAD_RESPONSE | jq -r '.video.id')
else
    VIDEO_ID=$(echo $UPLOAD_RESPONSE | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
fi

if [ -z "$VIDEO_ID" ] || [ "$VIDEO_ID" = "null" ]; then
    echo "❌ Failed to extract video ID from response"
    echo "Response: $UPLOAD_RESPONSE"
    exit 1
fi

echo "✅ Upload URL generated with video ID: $VIDEO_ID"

# 3. Test annotation fetch (should return empty array)
echo ""
echo "3️⃣ Testing annotation fetch..."
echo "-------------------------------------------"
ANNOTATION_RESPONSE=$(curl -s -X GET "$API_URL/videos/$VIDEO_ID/annotations")

echo "Annotation response: $ANNOTATION_RESPONSE"

# Check if the response is valid
if echo "$ANNOTATION_RESPONSE" | grep -q '"success":true'; then
    echo "✅ Annotation fetch successful"
else
    echo "❌ Annotation fetch failed"
    echo "Response: $ANNOTATION_RESPONSE"
    exit 1
fi

# 4. Test annotation creation
echo ""
echo "4️⃣ Testing annotation creation..."
echo "-------------------------------------------"
CREATE_ANNOTATION_RESPONSE=$(curl -s -X POST "$API_URL/videos/$VIDEO_ID/annotations" \
  -H "Content-Type: application/json" \
  -d '{
    "startTime": 0,
    "endTime": 5,
    "label": "Test Annotation",
    "confidence": 0.95,
    "notes": "This is a test annotation"
  }')

echo "Create annotation response: $CREATE_ANNOTATION_RESPONSE"

# Check if creation was successful
if echo "$CREATE_ANNOTATION_RESPONSE" | grep -q '"success":true'; then
    echo "✅ Annotation creation successful"
else
    echo "❌ Annotation creation failed"
    echo "Response: $CREATE_ANNOTATION_RESPONSE"
    exit 1
fi

# 5. Test annotation fetch again (should return the created annotation)
echo ""
echo "5️⃣ Testing annotation fetch after creation..."
echo "-------------------------------------------"
ANNOTATION_RESPONSE_2=$(curl -s -X GET "$API_URL/videos/$VIDEO_ID/annotations")

echo "Annotation response after creation: $ANNOTATION_RESPONSE_2"

# Check if the response contains the annotation
if echo "$ANNOTATION_RESPONSE_2" | grep -q '"success":true'; then
    echo "✅ Annotation fetch after creation successful"
else
    echo "❌ Annotation fetch after creation failed"
    echo "Response: $ANNOTATION_RESPONSE_2"
    exit 1
fi

echo ""
echo "🎉 All tests passed! The complete flow is working correctly."
echo "Session ID: $SESSION_ID"
echo "Video ID: $VIDEO_ID" 