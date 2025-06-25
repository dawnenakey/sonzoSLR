#!/bin/bash

# Test script for Spokhand upload URL generation
API_URL="https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod"

echo "🧪 Testing Spokhand Upload URL Generation"
echo "=========================================="

# First, create a session to get a session ID
echo ""
echo "1️⃣ Creating a test session..."
echo "-------------------------------------------"
SESSION_RESPONSE=$(curl -s -X POST "$API_URL/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Upload Test Session",
    "description": "Testing upload URL generation"
  }')

echo "Session response: $SESSION_RESPONSE"

# Extract session ID from response using jq if available, otherwise use grep
if command -v jq &> /dev/null; then
    SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.session_id')
else
    SESSION_ID=$(echo $SESSION_RESPONSE | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
fi

if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
    echo "❌ Failed to create session or extract session ID"
    echo "Response: $SESSION_RESPONSE"
    exit 1
fi

echo "✅ Created session with ID: $SESSION_ID"

# Test 2: Generate upload URL
echo ""
echo "2️⃣ Testing POST /sessions/{sessionId}/upload-video (Generate Upload URL)"
echo "------------------------------------------------------------------------"
curl -X POST "$API_URL/sessions/$SESSION_ID/upload-video" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test_video.mp4",
    "contentType": "video/mp4"
  }' \
  -w "\n\nHTTP Status: %{http_code}\nResponse Time: %{time_total}s\n"

echo ""
echo "✅ Upload URL Test Complete!" 