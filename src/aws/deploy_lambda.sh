#!/bin/bash

# Deploy Lambda function with updated code and dependencies
echo "Deploying Lambda function..."

# Update the Lambda function code
aws lambda update-function-code \
    --function-name spokhand-processor \
    --zip-file fileb://lambda_function.zip \
    --region us-east-1

echo "Lambda function code updated successfully!"

# Set environment variables for DynamoDB
aws lambda update-function-configuration \
    --function-name spokhand-processor \
    --environment Variables="{DYNAMODB_TABLE=spokhand-data-collection}" \
    --region us-east-1

echo "Environment variables updated!"

echo "Deployment complete! You can now test the API endpoint." 