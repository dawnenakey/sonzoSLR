#!/bin/bash

echo "🔍 Debugging Spokhand Lambda and DynamoDB"
echo "=========================================="

echo "1. Checking DynamoDB table..."
aws dynamodb describe-table --table-name spokhand-data-collection --output json 2>/dev/null || echo "DynamoDB table not found or access denied"

echo ""
echo "2. Testing Lambda function directly..."
aws lambda invoke --function-name spokhand-processor --payload '{"httpMethod":"POST","path":"/sessions","body":"{\"name\":\"Debug Test\",\"description\":\"Debug test\"}"}' debug_response.json
if [ -f debug_response.json ]; then
cat debug_response.json
rm debug_response.json
else
    echo "No response file generated"
fi

echo ""
echo "3. Checking Lambda function configuration..."
aws lambda get-function-configuration --function-name spokhand-processor --output json 2>/dev/null || echo "Lambda function not found or access denied"

echo ""
echo "4. Checking Lambda function logs..."
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/spokhand-processor" --output json 2>/dev/null || echo "No log groups found" 