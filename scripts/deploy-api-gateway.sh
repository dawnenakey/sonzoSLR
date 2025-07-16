#!/bin/bash

# Deploy API Gateway for SpokHand SLR ASL-LEX Service
# This script sets up API Gateway with Lambda integration for all ASL-LEX endpoints

set -e

# Configuration
STACK_NAME="spokhand-asl-lex-api"
ENVIRONMENT=${1:-dev}
REGION=${2:-us-east-1}

echo "🚀 Deploying SpokHand ASL-LEX API Gateway..."
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Stack Name: $STACK_NAME"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Create deployment package
echo "📦 Creating Lambda deployment package..."

# Create temporary directory for deployment
TEMP_DIR=$(mktemp -d)
DEPLOYMENT_PACKAGE="$TEMP_DIR/asl_lex_service.zip"

# Copy the ASL-LEX service file
cp src/asl_lex_service.py "$TEMP_DIR/"

# Install dependencies to the temp directory
pip install -r requirements.txt -t "$TEMP_DIR/" --quiet

# Create the deployment package
cd "$TEMP_DIR"
zip -r "$DEPLOYMENT_PACKAGE" . -q
cd - > /dev/null

echo "✅ Deployment package created: $DEPLOYMENT_PACKAGE"

# Deploy CloudFormation stack
echo "🏗️ Deploying CloudFormation stack..."

aws cloudformation deploy \
    --template-file infrastructure/api-gateway-setup.yaml \
    --stack-name "$STACK_NAME-$ENVIRONMENT" \
    --parameter-overrides Environment="$ENVIRONMENT" \
    --capabilities CAPABILITY_IAM \
    --region "$REGION"

# Get the API Gateway URL
API_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME-$ENVIRONMENT" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayUrl`].OutputValue' \
    --output text)

echo "✅ API Gateway deployed successfully!"
echo "🌐 API Gateway URL: $API_URL"

# Test the API endpoints
echo "🧪 Testing API endpoints..."

# Test sign types endpoint
echo "Testing GET /api/asl-lex/sign-types..."
curl -s "$API_URL/api/asl-lex/sign-types" | jq '.' || echo "❌ Sign types endpoint test failed"

# Test analytics endpoint
echo "Testing GET /api/asl-lex/analytics/sign-types..."
curl -s "$API_URL/api/asl-lex/analytics/sign-types" | jq '.' || echo "❌ Analytics endpoint test failed"

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Update your frontend API_BASE_URL to: $API_URL"
echo "2. Test the endpoints with your frontend application"
echo "3. Monitor the API Gateway in the AWS Console"
echo ""
echo "🔗 Useful links:"
echo "- API Gateway Console: https://console.aws.amazon.com/apigateway/"
echo "- CloudFormation Console: https://console.aws.amazon.com/cloudformation/"
echo "- Lambda Console: https://console.aws.amazon.com/lambda/"
echo ""
echo "📊 To monitor costs:"
echo "- CloudWatch Metrics: https://console.aws.amazon.com/cloudwatch/"
echo "- AWS Cost Explorer: https://console.aws.amazon.com/cost-management/" 