#!/bin/bash

echo "🔍 Debugging Spokhand Lambda and DynamoDB"
echo "=========================================="

echo "1. Checking DynamoDB table..."
aws dynamodb describe-table --table-name spokhand-data-collection --output json

echo ""
echo "2. Testing Lambda function directly..."
aws lambda invoke --function-name spokhand-processor --payload '{"httpMethod":"POST","path":"/sessions","body":"{\"name\":\"Debug Test\",\"description\":\"Debug test\"}"}' debug_response.json
cat debug_response.json
rm debug_response.json

echo ""
echo "3. Checking Lambda function configuration..."
aws lambda get-function-configuration --function-name spokhand-processor --output json 