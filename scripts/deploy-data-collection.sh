#!/bin/bash

# SpokHand SLR Remote Data Collection Deployment Script
set -e

echo "🚀 Deploying SpokHand Remote Data Collection Infrastructure..."

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-us-east-1}
STACK_NAME="spokhand-data-collection-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check AWS CLI installation
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

print_status "Deploying CloudFormation stack: ${STACK_NAME}"

# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file aws-deployment/remote-data-collection.yml \
    --stack-name ${STACK_NAME} \
    --parameter-overrides Environment=${ENVIRONMENT} \
    --capabilities CAPABILITY_IAM \
    --region ${AWS_REGION}

if [ $? -eq 0 ]; then
    print_status "CloudFormation stack deployed successfully!"
    
    # Get stack outputs
    print_status "Retrieving stack outputs..."
    
    S3_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${AWS_REGION} \
        --query 'Stacks[0].Outputs[?OutputKey==`VideoStorageBucketName`].OutputValue' \
        --output text)
    
    DYNAMODB_TABLE=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${AWS_REGION} \
        --query 'Stacks[0].Outputs[?OutputKey==`DataCollectionTableName`].OutputValue' \
        --output text)
    
    API_ENDPOINT=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${AWS_REGION} \
        --query 'Stacks[0].Outputs[?OutputKey==`APIEndpoint`].OutputValue' \
        --output text)
    
    print_status "Deployment completed successfully!"
    echo ""
    echo "📋 Deployment Summary:"
    echo "  Environment: ${ENVIRONMENT}"
    echo "  Region: ${AWS_REGION}"
    echo "  S3 Bucket: ${S3_BUCKET}"
    echo "  DynamoDB Table: ${DYNAMODB_TABLE}"
    echo "  API Endpoint: ${API_ENDPOINT}"
    echo ""
    
    # Create environment file for the data collection service
    cat > microservices/data-collection-service/.env << EOF
AWS_REGION=${AWS_REGION}
S3_BUCKET=${S3_BUCKET}
DYNAMODB_TABLE=${DYNAMODB_TABLE}
API_ENDPOINT=${API_ENDPOINT}
EOF
    
    print_status "Environment file created: microservices/data-collection-service/.env"
    
    # Instructions for next steps
    echo ""
    echo "🎯 Next Steps:"
    echo "1. Deploy the data collection service:"
    echo "   cd microservices/data-collection-service"
    echo "   pip install -r requirements.txt"
    echo "   python main.py"
    echo ""
    echo "2. Update frontend environment variables:"
    echo "   Add REACT_APP_DATA_COLLECTION_API=${API_ENDPOINT} to frontend/.env"
    echo ""
    echo "3. Test the remote data collection system"
    
else
    print_error "CloudFormation deployment failed!"
    exit 1
fi 