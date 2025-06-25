#!/bin/bash

# Script to fix API Gateway routes for Spokhand annotation endpoints
API_ID="qt8f7grhb5"
STAGE_NAME="prod"

echo "🔧 Fixing API Gateway routes for Spokhand annotation endpoints"
echo "=============================================================="

# Get all resources and find the videos resource
echo "1️⃣ Getting all API Gateway resources..."
RESOURCES=$(aws apigateway get-resources --rest-api-id $API_ID)

# Extract the videos resource ID
VIDEOS_RESOURCE_ID=$(echo "$RESOURCES" | grep -A 5 -B 5 '"path": "/videos"' | grep '"id"' | head -1 | sed 's/.*"id": "\([^"]*\)".*/\1/')

if [ -z "$VIDEOS_RESOURCE_ID" ]; then
    echo "❌ Could not find /videos resource. Creating it..."
    ROOT_RESOURCE_ID=$(echo "$RESOURCES" | grep -A 5 -B 5 '"path": "/"' | grep '"id"' | head -1 | sed 's/.*"id": "\([^"]*\)".*/\1/')
    VIDEOS_RESOURCE_ID=$(aws apigateway create-resource \
      --rest-api-id $API_ID \
      --parent-id $ROOT_RESOURCE_ID \
      --path-part "videos" \
      --query 'id' \
      --output text)
fi

echo "Videos resource ID: $VIDEOS_RESOURCE_ID"

# Get or create the {videoId} resource
echo "2️⃣ Getting or creating {videoId} resource..."
VIDEO_ID_RESOURCE_ID=$(echo "$RESOURCES" | grep -A 5 -B 5 '"path": "/videos/{videoId}"' | grep '"id"' | head -1 | sed 's/.*"id": "\([^"]*\)".*/\1/')

if [ -z "$VIDEO_ID_RESOURCE_ID" ]; then
    echo "Creating {videoId} resource..."
    VIDEO_ID_RESOURCE_ID=$(aws apigateway create-resource \
      --rest-api-id $API_ID \
      --parent-id $VIDEOS_RESOURCE_ID \
      --path-part "{videoId}" \
      --query 'id' \
      --output text)
fi

echo "Video ID resource ID: $VIDEO_ID_RESOURCE_ID"

# Get or create the annotations resource
echo "3️⃣ Getting or creating annotations resource..."
ANNOTATIONS_RESOURCE_ID=$(echo "$RESOURCES" | grep -A 5 -B 5 '"path": "/videos/{videoId}/annotations"' | grep '"id"' | head -1 | sed 's/.*"id": "\([^"]*\)".*/\1/')

if [ -z "$ANNOTATIONS_RESOURCE_ID" ]; then
    echo "Creating annotations resource..."
    ANNOTATIONS_RESOURCE_ID=$(aws apigateway create-resource \
      --rest-api-id $API_ID \
      --parent-id $VIDEO_ID_RESOURCE_ID \
      --path-part "annotations" \
      --query 'id' \
      --output text)
fi

echo "Annotations resource ID: $ANNOTATIONS_RESOURCE_ID"

# Get the Lambda function ARN
echo "4️⃣ Getting Lambda function ARN..."
LAMBDA_ARN=$(aws lambda get-function --function-name spokhand-processor --query 'Configuration.FunctionArn' --output text)
echo "Lambda ARN: $LAMBDA_ARN"

# Add GET method for /videos/{videoId}/annotations
echo "5️⃣ Adding GET method for annotations..."
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method GET \
  --authorization-type NONE

# Add POST method for /videos/{videoId}/annotations
echo "6️⃣ Adding POST method for annotations..."
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method POST \
  --authorization-type NONE

# Add Lambda integration for GET annotations
echo "7️⃣ Adding Lambda integration for GET annotations..."
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method GET \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations

# Add Lambda integration for POST annotations
echo "8️⃣ Adding Lambda integration for POST annotations..."
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $ANNOTATIONS_RESOURCE_ID \
  --http-method POST \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations

# Deploy the API
echo "9️⃣ Deploying API Gateway..."
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name $STAGE_NAME

echo ""
echo "✅ API Gateway fix complete!"
echo "The following endpoints should now be available:"
echo "  GET  /videos/{videoId}/annotations"
echo "  POST /videos/{videoId}/annotations"
echo ""
echo "API URL: https://$API_ID.execute-api.us-east-1.amazonaws.com/$STAGE_NAME" 