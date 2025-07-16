#!/bin/bash

# Add ASL-LEX endpoints to existing API Gateway
# This script adds new endpoints without overwriting existing ones

set -e

# Configuration - UPDATE THESE TO MATCH YOUR EXISTING SETUP
API_ID="qt8f7grhb5"  # Your existing API Gateway ID
STAGE_NAME="prod"     # Your existing stage name
LAMBDA_FUNCTION_NAME="spokhand-processor"  # Your existing Lambda function name

echo "🔧 Adding ASL-LEX endpoints to existing API Gateway..."
echo "API Gateway ID: $API_ID"
echo "Stage Name: $STAGE_NAME"
echo "Lambda Function: $LAMBDA_FUNCTION_NAME"
echo "=================================================="

# Get the root resource ID
echo "1️⃣ Getting root resource ID..."
ROOT_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text)
echo "Root resource ID: $ROOT_RESOURCE_ID"

# Get Lambda function ARN
echo "2️⃣ Getting Lambda function ARN..."
LAMBDA_ARN=$(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --query 'Configuration.FunctionArn' --output text)
echo "Lambda ARN: $LAMBDA_ARN"

# Create /api resource
echo "3️⃣ Creating /api resource..."
API_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_RESOURCE_ID \
  --path-part "api" \
  --query 'id' \
  --output text)
echo "API resource ID: $API_RESOURCE_ID"

# Create /api/asl-lex resource
echo "4️⃣ Creating /api/asl-lex resource..."
ASL_LEX_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $API_RESOURCE_ID \
  --path-part "asl-lex" \
  --query 'id' \
  --output text)
echo "ASL-LEX resource ID: $ASL_LEX_RESOURCE_ID"

# Create /api/asl-lex/signs resource
echo "5️⃣ Creating /api/asl-lex/signs resource..."
SIGNS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ASL_LEX_RESOURCE_ID \
  --path-part "signs" \
  --query 'id' \
  --output text)
echo "Signs resource ID: $SIGNS_RESOURCE_ID"

# Create /api/asl-lex/sign-types resource
echo "6️⃣ Creating /api/asl-lex/sign-types resource..."
SIGN_TYPES_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ASL_LEX_RESOURCE_ID \
  --path-part "sign-types" \
  --query 'id' \
  --output text)
echo "Sign types resource ID: $SIGN_TYPES_RESOURCE_ID"

# Create /api/asl-lex/sign-types/custom resource
echo "7️⃣ Creating /api/asl-lex/sign-types/custom resource..."
CUSTOM_SIGN_TYPES_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $SIGN_TYPES_RESOURCE_ID \
  --path-part "custom" \
  --query 'id' \
  --output text)
echo "Custom sign types resource ID: $CUSTOM_SIGN_TYPES_RESOURCE_ID"

# Create /api/asl-lex/signs/batch-update-type resource
echo "8️⃣ Creating /api/asl-lex/signs/batch-update-type resource..."
BATCH_UPDATE_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $SIGNS_RESOURCE_ID \
  --path-part "batch-update-type" \
  --query 'id' \
  --output text)
echo "Batch update resource ID: $BATCH_UPDATE_RESOURCE_ID"

# Create /api/asl-lex/validate-asl-sign resource
echo "9️⃣ Creating /api/asl-lex/validate-asl-sign resource..."
VALIDATE_SIGN_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ASL_LEX_RESOURCE_ID \
  --path-part "validate-asl-sign" \
  --query 'id' \
  --output text)
echo "Validate sign resource ID: $VALIDATE_SIGN_RESOURCE_ID"

# Create /api/asl-lex/analytics resource
echo "🔟 Creating /api/asl-lex/analytics resource..."
ANALYTICS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ASL_LEX_RESOURCE_ID \
  --path-part "analytics" \
  --query 'id' \
  --output text)
echo "Analytics resource ID: $ANALYTICS_RESOURCE_ID"

# Create /api/asl-lex/analytics/sign-types resource
echo "1️⃣1️⃣ Creating /api/asl-lex/analytics/sign-types resource..."
SIGN_TYPES_ANALYTICS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ANALYTICS_RESOURCE_ID \
  --path-part "sign-types" \
  --query 'id' \
  --output text)
echo "Sign types analytics resource ID: $SIGN_TYPES_ANALYTICS_RESOURCE_ID"

# Create /api/asl-lex/bulk-upload resource
echo "1️⃣2️⃣ Creating /api/asl-lex/bulk-upload resource..."
BULK_UPLOAD_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ASL_LEX_RESOURCE_ID \
  --path-part "bulk-upload" \
  --query 'id' \
  --output text)
echo "Bulk upload resource ID: $BULK_UPLOAD_RESOURCE_ID"

# Create /api/asl-lex/bulk-upload/jobs resource
echo "1️⃣3️⃣ Creating /api/asl-lex/bulk-upload/jobs resource..."
JOBS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $BULK_UPLOAD_RESOURCE_ID \
  --path-part "jobs" \
  --query 'id' \
  --output text)
echo "Jobs resource ID: $JOBS_RESOURCE_ID"

# Create /api/asl-lex/bulk-upload/template resource
echo "1️⃣4️⃣ Creating /api/asl-lex/bulk-upload/template resource..."
TEMPLATE_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $BULK_UPLOAD_RESOURCE_ID \
  --path-part "template" \
  --query 'id' \
  --output text)
echo "Template resource ID: $TEMPLATE_RESOURCE_ID"

# Create /api/asl-lex/statistics resource
echo "1️⃣5️⃣ Creating /api/asl-lex/statistics resource..."
STATISTICS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ASL_LEX_RESOURCE_ID \
  --path-part "statistics" \
  --query 'id' \
  --output text)
echo "Statistics resource ID: $STATISTICS_RESOURCE_ID"

echo ""
echo "🔗 Adding HTTP methods and Lambda integrations..."
echo "================================================"

# Function to add method and integration
add_method_and_integration() {
    local resource_id=$1
    local http_method=$2
    local path_name=$3
    
    echo "Adding $http_method method to $path_name..."
    
    # Add method
    aws apigateway put-method \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method $http_method \
      --authorization-type NONE
    
    # Add Lambda integration
    aws apigateway put-integration \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method $http_method \
      --type AWS_PROXY \
      --integration-http-method POST \
      --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations
}

# Add methods and integrations for all endpoints
add_method_and_integration $SIGNS_RESOURCE_ID "GET" "/api/asl-lex/signs"
add_method_and_integration $SIGNS_RESOURCE_ID "POST" "/api/asl-lex/signs"
add_method_and_integration $SIGN_TYPES_RESOURCE_ID "GET" "/api/asl-lex/sign-types"
add_method_and_integration $SIGN_TYPES_RESOURCE_ID "POST" "/api/asl-lex/sign-types"
add_method_and_integration $CUSTOM_SIGN_TYPES_RESOURCE_ID "GET" "/api/asl-lex/sign-types/custom"
add_method_and_integration $BATCH_UPDATE_RESOURCE_ID "POST" "/api/asl-lex/signs/batch-update-type"
add_method_and_integration $VALIDATE_SIGN_RESOURCE_ID "POST" "/api/asl-lex/validate-asl-sign"
add_method_and_integration $SIGN_TYPES_ANALYTICS_RESOURCE_ID "GET" "/api/asl-lex/analytics/sign-types"
add_method_and_integration $BULK_UPLOAD_RESOURCE_ID "POST" "/api/asl-lex/bulk-upload"
add_method_and_integration $JOBS_RESOURCE_ID "GET" "/api/asl-lex/bulk-upload/jobs"
add_method_and_integration $TEMPLATE_RESOURCE_ID "GET" "/api/asl-lex/bulk-upload/template"
add_method_and_integration $STATISTICS_RESOURCE_ID "GET" "/api/asl-lex/statistics"

# Deploy the API
echo ""
echo "🚀 Deploying API Gateway..."
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name $STAGE_NAME

echo ""
echo "✅ ASL-LEX endpoints added successfully!"
echo ""
echo "📋 New endpoints available:"
echo "  GET    /api/asl-lex/signs"
echo "  POST   /api/asl-lex/signs"
echo "  GET    /api/asl-lex/sign-types"
echo "  POST   /api/asl-lex/sign-types"
echo "  GET    /api/asl-lex/sign-types/custom"
echo "  POST   /api/asl-lex/signs/batch-update-type"
echo "  POST   /api/asl-lex/validate-asl-sign"
echo "  GET    /api/asl-lex/analytics/sign-types"
echo "  POST   /api/asl-lex/bulk-upload"
echo "  GET    /api/asl-lex/bulk-upload/jobs"
echo "  GET    /api/asl-lex/bulk-upload/template"
echo "  GET    /api/asl-lex/statistics"
echo ""
echo "🌐 API URL: https://$API_ID.execute-api.us-east-1.amazonaws.com/$STAGE_NAME"
echo ""
echo "📝 Next steps:"
echo "1. Update your Lambda function to handle the new ASL-LEX endpoints"
echo "2. Test the endpoints with your frontend application"
echo "3. Monitor the API Gateway in the AWS Console" 