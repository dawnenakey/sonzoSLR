#!/bin/bash

# Create DynamoDB table for data collection
echo "Creating DynamoDB table..."

aws dynamodb create-table \
    --table-name spokhand-data-collection \
    --attribute-definitions AttributeName=sessionId,AttributeType=S \
    --key-schema AttributeName=sessionId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

echo "DynamoDB table created successfully!"

# Wait for table to be active
echo "Waiting for table to be active..."
aws dynamodb wait table-exists \
    --table-name spokhand-data-collection \
    --region us-east-1

echo "Table is now active and ready to use!" 