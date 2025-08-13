#!/bin/bash

# Test ASL-LEX endpoints to see which ones are working
# This script will help identify which endpoints are missing

set -e

API_BASE_URL="https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod"

echo "🧪 Testing ASL-LEX Endpoints"
echo "============================="
echo "API Base URL: $API_BASE_URL"
echo ""

# Function to test an endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local description=$3
    
    echo "Testing $method $endpoint ($description)..."
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" "$API_BASE_URL$endpoint" -o /tmp/response.json)
    else
        response=$(curl -s -w "%{http_code}" -X POST "$API_BASE_URL$endpoint" -H "Content-Type: application/json" -d '{}' -o /tmp/response.json)
    fi
    
    if [ "$response" = "200" ] || [ "$response" = "201" ]; then
        echo "✅ $method $endpoint - SUCCESS (HTTP $response)"
    else
        echo "❌ $method $endpoint - FAILED (HTTP $response)"
        if [ -f /tmp/response.json ]; then
            echo "   Response: $(cat /tmp/response.json)"
        fi
    fi
    echo ""
}

# Test all expected endpoints
echo "🔍 Testing Available Endpoints:"
test_endpoint "GET" "/api/asl-lex/signs" "List signs"
test_endpoint "POST" "/api/asl-lex/signs" "Create sign"
test_endpoint "GET" "/api/asl-lex/sign-types" "List sign types"
test_endpoint "POST" "/api/asl-lex/sign-types" "Create sign type"
test_endpoint "GET" "/api/asl-lex/sign-types/custom" "List custom sign types"
test_endpoint "POST" "/api/asl-lex/signs/batch-update-type" "Batch update sign types"
test_endpoint "POST" "/api/asl-lex/validate-asl-sign" "Validate ASL sign"
test_endpoint "GET" "/api/asl-lex/analytics/sign-types" "Get sign type analytics"
test_endpoint "POST" "/api/asl-lex/bulk-upload" "Bulk upload"
test_endpoint "GET" "/api/asl-lex/bulk-upload/jobs" "List bulk upload jobs"

echo "🔍 Testing Missing Endpoints (should fail):"
test_endpoint "GET" "/api/asl-lex/statistics" "Get statistics"
test_endpoint "GET" "/api/asl-lex/bulk-upload/template" "Get bulk upload template"
test_endpoint "GET" "/api/asl-lex/bulk-upload/jobs/test-job" "Get specific job"
test_endpoint "POST" "/api/asl-lex/bulk-upload/jobs/test-job/cancel" "Cancel job"
test_endpoint "POST" "/api/asl-lex/upload-video-with-metadata" "Upload video with metadata"

echo "📊 Summary:"
echo "✅ Working endpoints should show SUCCESS"
echo "❌ Missing endpoints should show FAILED (404 or 500)"
echo ""
echo "If you see many FAILED responses, you need to add the missing endpoints to your API Gateway."
echo "Use the deployment guide: ./scripts/deploy-missing-endpoints.sh" 