#!/bin/bash

# Test script for Spokhand API Gateway endpoints
API_URL="https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod"

echo "🧪 Testing Spokhand API Gateway Endpoints"
echo "=========================================="

# Test 1: Create a new session
echo ""
echo "1️⃣ Testing POST /sessions (Create Session)"
echo "-------------------------------------------"
curl -X POST "$API_URL/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Session from CLI",
    "description": "Testing the API from Mac terminal"
  }' \
  -w "\n\nHTTP Status: %{http_code}\nResponse Time: %{time_total}s\n"

# Test 2: Get sessions (if we have any)
echo ""
echo "2️⃣ Testing GET /sessions (List Sessions)"
echo "----------------------------------------"
curl -X GET "$API_URL/sessions" \
  -H "Content-Type: application/json" \
  -w "\n\nHTTP Status: %{http_code}\nResponse Time: %{time_total}s\n"

echo ""
echo "✅ API Testing Complete!" 