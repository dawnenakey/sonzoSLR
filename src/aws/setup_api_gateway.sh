#!/bin/bash

# Script to add missing API Gateway routes for Spokhand annotation endpoints
API_ID="qt8f7grhb5"
STAGE_NAME="prod"

echo "🔧 Setting up API Gateway routes for Spokhand annotation endpoints"
echo "=================================================================="

# Get the root resource ID
echo "1️⃣ Getting root resource ID..."
ROOT_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text)
echo "Root resource ID: $ROOT_RESOURCE_ID"

# Get the sessions resource ID
echo "2️⃣ Getting sessions resource ID..."
SESSIONS_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/sessions`].id' --output text)
echo "Sessions resource ID: $SESSIONS_RESOURCE_ID"

# Create videos resource
echo "3️⃣ Creating videos resource..."
VIDEOS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_RESOURCE_ID \
  --path-part "videos" \
  --query 'id' \
  --output text)
echo "Videos resource ID: $VIDEOS_RESOURCE_ID"

# Create {videoId} resource
echo "4️⃣ Creating {videoId} resource..."
VIDEO_ID_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $VIDEOS_RESOURCE_ID \
  --path-part "{videoId}" \
  --query 'id' \
  --output text)
echo "Video ID resource ID: $VIDEO_ID_RESOURCE_ID"

# Create annotations resource
echo "5️⃣ Creating annotations resource..."
ANNOTATIONS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $VIDEO_ID_RESOURCE_ID \
  --path-part "annotations" \
  --query 'id' \
  --output text)
echo "Annotations resource ID: $ANNOTATIONS_RESOURCE_ID"

# Add GET method for /videos/{videoId}/annotations
echo "6️⃣ Adding GET method for annotations..."
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method GET \
  --authorization-type NONE

# Add POST method for /videos/{videoId}/annotations
echo "7️⃣ Adding POST method for annotations..."
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method POST \
  --authorization-type NONE

# Get the Lambda function ARN
echo "8️⃣ Getting Lambda function ARN..."
LAMBDA_ARN=$(aws lambda get-function --function-name spokhand-processor --query 'Configuration.FunctionArn' --output text)
echo "Lambda ARN: $LAMBDA_ARN"

# Add Lambda integration for GET annotations
echo "9️⃣ Adding Lambda integration for GET annotations..."
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method GET \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations

# Add Lambda integration for POST annotations
echo "🔟 Adding Lambda integration for POST annotations..."
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method POST \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations

# Deploy the API
echo "1️⃣1️⃣ Deploying API Gateway..."
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name $STAGE_NAME

echo ""
echo "✅ API Gateway setup complete!"
echo "The following endpoints are now available:"
echo "  GET  /videos/{videoId}/annotations"
echo "  POST /videos/{videoId}/annotations"
echo ""
echo "API URL: https://$API_ID.execute-api.us-east-1.amazonaws.com/$STAGE_NAME" 