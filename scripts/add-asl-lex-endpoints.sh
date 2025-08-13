#!/bin/bash

# Add missing ASL-LEX endpoints to existing API Gateway
# This script adds the missing endpoints that the frontend expects

set -e

# Configuration - UPDATE THESE TO MATCH YOUR SETUP
API_ID="qt8f7grhb5"  # Your existing API Gateway ID
STAGE_NAME="prod"     # Your existing stage name

echo "🔧 Adding missing ASL-LEX endpoints to API Gateway..."
echo "API Gateway ID: $API_ID"
echo "Stage Name: $STAGE_NAME"
echo "=================================================="

# Get the root resource ID
ROOT_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text)
echo "Root Resource ID: $ROOT_RESOURCE_ID"

# Get the /api resource ID
API_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api`].id' --output text)
if [ -z "$API_RESOURCE_ID" ]; then
    echo "Creating /api resource..."
    API_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $ROOT_RESOURCE_ID \
        --path-part "api" \
        --query 'id' --output text)
fi

# Get the /api/asl-lex resource ID
ASL_LEX_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api/asl-lex`].id' --output text)
if [ -z "$ASL_LEX_RESOURCE_ID" ]; then
    echo "Creating /api/asl-lex resource..."
    ASL_LEX_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $API_RESOURCE_ID \
        --path-part "asl-lex" \
        --query 'id' --output text)
fi

# Get the /api/asl-lex/bulk-upload resource ID
BULK_UPLOAD_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api/asl-lex/bulk-upload`].id' --output text)
if [ -z "$BULK_UPLOAD_RESOURCE_ID" ]; then
    echo "Creating /api/asl-lex/bulk-upload resource..."
    BULK_UPLOAD_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $ASL_LEX_RESOURCE_ID \
        --path-part "bulk-upload" \
        --query 'id' --output text)
fi

# Get the /api/asl-lex/bulk-upload/jobs resource ID
JOBS_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/api/asl-lex/bulk-upload/jobs`].id' --output text)
if [ -z "$JOBS_RESOURCE_ID" ]; then
    echo "Creating /api/asl-lex/bulk-upload/jobs resource..."
    JOBS_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $BULK_UPLOAD_RESOURCE_ID \
        --path-part "jobs" \
        --query 'id' --output text)
fi

echo ""
echo "🔗 Adding missing endpoints..."
echo "=================================="

# Add /api/asl-lex/statistics endpoint
echo "Adding GET /api/asl-lex/statistics..."
STATISTICS_RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ASL_LEX_RESOURCE_ID \
    --path-part "statistics" \
    --query 'id' --output text)

aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $STATISTICS_RESOURCE_ID \
    --http-method GET \
    --authorization-type NONE

aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $STATISTICS_RESOURCE_ID \
    --http-method GET \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:spokhand-asl-lex-service-prod/invocations"

# Add /api/asl-lex/bulk-upload/template endpoint
echo "Adding GET /api/asl-lex/bulk-upload/template..."
TEMPLATE_RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $BULK_UPLOAD_RESOURCE_ID \
    --path-part "template" \
    --query 'id' --output text)

aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $TEMPLATE_RESOURCE_ID \
    --http-method GET \
    --authorization-type NONE

aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $TEMPLATE_RESOURCE_ID \
    --http-method GET \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:spokhand-asl-lex-service-prod/invocations"

# Add /api/asl-lex/bulk-upload/jobs/{jobId} endpoint
echo "Adding GET /api/asl-lex/bulk-upload/jobs/{jobId}..."
JOB_RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $JOBS_RESOURCE_ID \
    --path-part "{jobId}" \
    --query 'id' --output text)

aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $JOB_RESOURCE_ID \
    --http-method GET \
    --authorization-type NONE \
    --request-parameters '{"method.request.path.jobId": true}'

aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $JOB_RESOURCE_ID \
    --http-method GET \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:spokhand-asl-lex-service-prod/invocations"

# Add /api/asl-lex/bulk-upload/jobs/{jobId}/cancel endpoint
echo "Adding POST /api/asl-lex/bulk-upload/jobs/{jobId}/cancel..."
CANCEL_RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $JOB_RESOURCE_ID \
    --path-part "cancel" \
    --query 'id' --output text)

aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $CANCEL_RESOURCE_ID \
    --http-method POST \
    --authorization-type NONE \
    --request-parameters '{"method.request.path.jobId": true}'

aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $CANCEL_RESOURCE_ID \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:spokhand-asl-lex-service-prod/invocations"

# Add /api/asl-lex/upload-video-with-metadata endpoint
echo "Adding POST /api/asl-lex/upload-video-with-metadata..."
UPLOAD_VIDEO_RESOURCE_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ASL_LEX_RESOURCE_ID \
    --path-part "upload-video-with-metadata" \
    --query 'id' --output text)

aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $UPLOAD_VIDEO_RESOURCE_ID \
    --http-method POST \
    --authorization-type NONE

aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $UPLOAD_VIDEO_RESOURCE_ID \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:YOUR_ACCOUNT_ID:function:spokhand-asl-lex-service-prod/invocations"

echo ""
echo "🚀 Deploying API Gateway changes..."
aws apigateway create-deployment \
    --rest-api-id $API_ID \
    --stage-name $STAGE_NAME \
    --description "Added missing ASL-LEX endpoints"

echo ""
echo "✅ Missing ASL-LEX endpoints added successfully!"
echo ""
echo "📋 Added endpoints:"
echo "  - GET /api/asl-lex/statistics"
echo "  - GET /api/asl-lex/bulk-upload/template"
echo "  - GET /api/asl-lex/bulk-upload/jobs/{jobId}"
echo "  - POST /api/asl-lex/bulk-upload/jobs/{jobId}/cancel"
echo "  - POST /api/asl-lex/upload-video-with-metadata"
echo ""
echo "🔗 Your API Gateway URL: https://${API_ID}.execute-api.us-east-1.amazonaws.com/${STAGE_NAME}"
echo ""
echo "⚠️  IMPORTANT: Update the Lambda ARN in this script with your actual Lambda function ARN"
echo "   Replace 'YOUR_ACCOUNT_ID' with your AWS account ID"
echo "   Replace 'spokhand-asl-lex-service-prod' with your actual Lambda function name" 