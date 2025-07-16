#!/bin/bash

# Fix CORS configuration for API Gateway
# This script updates the CORS settings to allow requests from the Amplify app

set -e

# Configuration - UPDATE THESE TO MATCH YOUR SETUP
API_ID="qt8f7grhb5"  # Your existing API Gateway ID
STAGE_NAME="prod"     # Your existing stage name

echo "🔧 Fixing CORS configuration for API Gateway..."
echo "API Gateway ID: $API_ID"
echo "Stage Name: $STAGE_NAME"
echo "=================================================="

# Get all resources
echo "1️⃣ Getting all API Gateway resources..."
RESOURCES=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[*].{id:id,path:path}' --output json)
echo "Found resources:"
echo "$RESOURCES" | jq '.[] | "\(.path) -> \(.id)"'

# Function to add CORS to a resource
add_cors_to_resource() {
    local resource_id=$1
    local path_name=$2
    
    echo "Adding CORS to resource: $path_name (ID: $resource_id)"
    
    # Add OPTIONS method for CORS preflight
    aws apigateway put-method \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --authorization-type NONE
    
    # Add CORS integration response
    aws apigateway put-integration \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --type MOCK \
      --request-templates '{"application/json":"{\"statusCode\": 200}"}'
    
    # Add method response
    aws apigateway put-method-response \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --status-code 200 \
      --response-parameters '{
        "method.response.header.Access-Control-Allow-Headers": true,
        "method.response.header.Access-Control-Allow-Methods": true,
        "method.response.header.Access-Control-Allow-Origin": true,
        "method.response.header.Access-Control-Allow-Credentials": true
      }'
    
    # Add integration response
    aws apigateway put-integration-response \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --status-code 200 \
      --response-parameters '{
        "method.response.header.Access-Control-Allow-Headers": "'\''Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'\''",
        "method.response.header.Access-Control-Allow-Methods": "'\''GET,POST,PUT,DELETE,OPTIONS'\''",
        "method.response.header.Access-Control-Allow-Origin": "'\''https://main.djz1od5v7st0v.amplifyapp.com'\''",
        "method.response.header.Access-Control-Allow-Credentials": "'\''true'\''"
      }'
}

# Function to update existing methods with CORS headers
update_method_cors() {
    local resource_id=$1
    local http_method=$2
    local path_name=$3
    
    echo "Updating CORS for $http_method method on $path_name"
    
    # Update method response to include CORS headers
    aws apigateway put-method-response \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method $http_method \
      --status-code 200 \
      --response-parameters '{
        "method.response.header.Access-Control-Allow-Origin": true,
        "method.response.header.Access-Control-Allow-Credentials": true
      }'
    
    # Update integration response to include CORS headers
    aws apigateway put-integration-response \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method $http_method \
      --status-code 200 \
      --response-parameters '{
        "method.response.header.Access-Control-Allow-Origin": "'\''https://main.djz1od5v7st0v.amplifyapp.com'\''",
        "method.response.header.Access-Control-Allow-Credentials": "'\''true'\''"
      }'
}

echo ""
echo "🔗 Adding CORS to all resources..."
echo "=================================="

# Add CORS to root resource
ROOT_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text)
add_cors_to_resource $ROOT_RESOURCE_ID "/"

# Add CORS to /videos resource
VIDEOS_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/videos`].id' --output text)
if [ ! -z "$VIDEOS_RESOURCE_ID" ]; then
    add_cors_to_resource $VIDEOS_RESOURCE_ID "/videos"
    update_method_cors $VIDEOS_RESOURCE_ID "GET" "/videos"
    update_method_cors $VIDEOS_RESOURCE_ID "POST" "/videos"
fi

# Add CORS to /sessions resource
SESSIONS_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/sessions`].id' --output text)
if [ ! -z "$SESSIONS_RESOURCE_ID" ]; then
    add_cors_to_resource $SESSIONS_RESOURCE_ID "/sessions"
    update_method_cors $SESSIONS_RESOURCE_ID "GET" "/sessions"
    update_method_cors $SESSIONS_RESOURCE_ID "POST" "/sessions"
fi

# Add CORS to /api/asl-lex resources (if they exist)
API_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api`].id' --output text)
if [ ! -z "$API_RESOURCE_ID" ]; then
    add_cors_to_resource $API_RESOURCE_ID "/api"
    
    ASL_LEX_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api/asl-lex`].id' --output text)
    if [ ! -z "$ASL_LEX_RESOURCE_ID" ]; then
        add_cors_to_resource $ASL_LEX_RESOURCE_ID "/api/asl-lex"
        
        # Add CORS to ASL-LEX sub-resources
        SIGNS_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api/asl-lex/signs`].id' --output text)
        if [ ! -z "$SIGNS_RESOURCE_ID" ]; then
            add_cors_to_resource $SIGNS_RESOURCE_ID "/api/asl-lex/signs"
            update_method_cors $SIGNS_RESOURCE_ID "GET" "/api/asl-lex/signs"
            update_method_cors $SIGNS_RESOURCE_ID "POST" "/api/asl-lex/signs"
        fi
        
        SIGN_TYPES_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api/asl-lex/sign-types`].id' --output text)
        if [ ! -z "$SIGN_TYPES_RESOURCE_ID" ]; then
            add_cors_to_resource $SIGN_TYPES_RESOURCE_ID "/api/asl-lex/sign-types"
            update_method_cors $SIGN_TYPES_RESOURCE_ID "GET" "/api/asl-lex/sign-types"
            update_method_cors $SIGN_TYPES_RESOURCE_ID "POST" "/api/asl-lex/sign-types"
        fi
    fi
fi

echo ""
echo "🚀 Deploying API Gateway changes..."
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name $STAGE_NAME \
  --description "CORS fix deployment"

echo ""
echo "✅ CORS configuration updated successfully!"
echo "Your API Gateway should now accept requests from:"
echo "  - https://main.djz1od5v7st0v.amplifyapp.com"
echo ""
echo "Test the fix by refreshing your application." 