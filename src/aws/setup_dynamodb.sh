#!/bin/bash

echo "🔧 Setting up DynamoDB table for Spokhand..."

TABLE_NAME="spokhand-data-collection"

# Check if table exists
if aws dynamodb describe-table --table-name $TABLE_NAME 2>/dev/null; then
    echo "✅ Table $TABLE_NAME already exists"
else
    echo "📦 Creating table $TABLE_NAME..."
    
    aws dynamodb create-table \
        --table-name $TABLE_NAME \
        --attribute-definitions \
            AttributeName=session_id,AttributeType=S \
        --key-schema \
            AttributeName=session_id,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region us-east-1
    
    echo "⏳ Waiting for table to be active..."
    aws dynamodb wait table-exists --table-name $TABLE_NAME
    echo "✅ Table $TABLE_NAME created successfully"
fi

echo "🔧 Setting Lambda environment variable..."
aws lambda update-function-configuration \
    --function-name spokhand-processor \
    --environment Variables="{DYNAMODB_TABLE=$TABLE_NAME}"

echo "✅ Setup complete!" 