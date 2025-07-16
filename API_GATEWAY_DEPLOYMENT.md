# API Gateway Deployment Guide

This guide explains how to deploy the ASL-LEX API endpoints using AWS API Gateway and Lambda.

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Make the deployment script executable
chmod +x scripts/deploy-api-gateway.sh

# Deploy to dev environment
./scripts/deploy-api-gateway.sh dev

# Deploy to production
./scripts/deploy-api-gateway.sh prod
```

### Option 2: Manual Deployment

```bash
# 1. Deploy the CloudFormation stack
aws cloudformation deploy \
    --template-file infrastructure/api-gateway-setup.yaml \
    --stack-name spokhand-asl-lex-api-dev \
    --parameter-overrides Environment=dev \
    --capabilities CAPABILITY_IAM

# 2. Get the API Gateway URL
aws cloudformation describe-stacks \
    --stack-name spokhand-asl-lex-api-dev \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayUrl`].OutputValue' \
    --output text
```

## 📋 Prerequisites

1. **AWS CLI installed and configured**
   ```bash
   aws --version
   aws configure
   ```

2. **Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **AWS permissions**
   - CloudFormation permissions
   - Lambda permissions
   - API Gateway permissions
   - DynamoDB permissions
   - S3 permissions

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   Lambda        │
│   (React)       │───▶│   (REST API)    │───▶│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   DynamoDB      │
                                              │   (Database)    │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   S3 Bucket     │
                                              │   (Videos)      │
                                              └─────────────────┘
```

## 🔗 API Endpoints

### Core Endpoints
- `GET /api/asl-lex/signs` - List all signs with filtering
- `POST /api/asl-lex/signs` - Create a new sign
- `GET /api/asl-lex/sign-types` - Get available sign types
- `POST /api/asl-lex/sign-types` - Add custom sign type
- `GET /api/asl-lex/sign-types/custom` - Get custom sign types

### Advanced Features
- `POST /api/asl-lex/signs/batch-update-type` - Batch update sign types
- `POST /api/asl-lex/validate-asl-sign` - Validate ASL sign data
- `GET /api/asl-lex/analytics/sign-types` - Get sign type analytics

### Bulk Upload
- `POST /api/asl-lex/bulk-upload` - Start bulk upload job
- `GET /api/asl-lex/bulk-upload/jobs` - List bulk upload jobs
- `GET /api/asl-lex/bulk-upload/template` - Download CSV template

### Statistics
- `GET /api/asl-lex/statistics` - Get system statistics

## 🔧 Configuration

### Environment Variables

The Lambda function uses these environment variables:

```bash
ENVIRONMENT=dev
DYNAMODB_TABLE=spokhand-asl-lex-dev
S3_BUCKET=spokhand-asl-lex-videos-dev
```

### CORS Configuration

API Gateway is configured with CORS headers:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Headers: Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token`
- `Access-Control-Allow-Methods: GET,POST,PUT,DELETE,OPTIONS`

## 🧪 Testing the API

### Test with curl

```bash
# Get sign types
curl -X GET "https://your-api-gateway-url.amazonaws.com/dev/api/asl-lex/sign-types"

# Get analytics
curl -X GET "https://your-api-gateway-url.amazonaws.com/dev/api/asl-lex/analytics/sign-types"

# Validate ASL sign
curl -X POST "https://your-api-gateway-url.amazonaws.com/dev/api/asl-lex/validate-asl-sign" \
  -H "Content-Type: application/json" \
  -d '{
    "gloss": "HELLO",
    "handshape": "B",
    "location": "neutral space",
    "sign_type": "isolated_sign"
  }'
```

### Test with JavaScript

```javascript
const API_BASE_URL = 'https://your-api-gateway-url.amazonaws.com/dev';

// Get sign types
fetch(`${API_BASE_URL}/api/asl-lex/sign-types`)
  .then(response => response.json())
  .then(data => console.log(data));

// Add custom sign type
fetch(`${API_BASE_URL}/api/asl-lex/sign-types`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    custom_type: 'classifier_construction',
    description: 'Classifier construction signs'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📊 Monitoring

### CloudWatch Metrics
- API Gateway metrics: Request count, latency, error rate
- Lambda metrics: Duration, errors, throttles
- DynamoDB metrics: Read/write capacity, throttles

### Logs
- API Gateway logs: `/aws/apigateway/spokhand-asl-lex-api`
- Lambda logs: `/aws/lambda/spokhand-asl-lex-service-dev`

### Cost Monitoring
- API Gateway: Pay per request
- Lambda: Pay per execution
- DynamoDB: Pay per request (on-demand)
- S3: Pay per GB stored

## 🔄 Updating the API

### Update Lambda Code

```bash
# Create new deployment package
zip -r lambda-deployment.zip src/ requirements.txt

# Update Lambda function
aws lambda update-function-code \
  --function-name spokhand-asl-lex-service-dev \
  --zip-file fileb://lambda-deployment.zip
```

### Update API Gateway

```bash
# Deploy new API Gateway configuration
aws cloudformation deploy \
  --template-file infrastructure/api-gateway-setup.yaml \
  --stack-name spokhand-asl-lex-api-dev \
  --capabilities CAPABILITY_IAM
```

## 🚨 Troubleshooting

### Common Issues

1. **CORS Errors**
   - Check API Gateway CORS configuration
   - Verify headers in Lambda response

2. **Lambda Timeout**
   - Increase timeout in CloudFormation template
   - Optimize Lambda function performance

3. **DynamoDB Errors**
   - Check IAM permissions
   - Verify table name in environment variables

4. **S3 Upload Issues**
   - Check S3 bucket permissions
   - Verify presigned URL generation

### Debug Commands

```bash
# Check Lambda logs
aws logs tail /aws/lambda/spokhand-asl-lex-service-dev --follow

# Test Lambda function directly
aws lambda invoke \
  --function-name spokhand-asl-lex-service-dev \
  --payload '{"httpMethod":"GET","path":"/api/asl-lex/sign-types"}' \
  response.json

# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix "/aws/apigateway"
```

## 🔐 Security Considerations

### IAM Permissions
- Least privilege principle
- Separate roles for different functions
- Regular permission audits

### Data Protection
- Encrypt data at rest (DynamoDB, S3)
- Encrypt data in transit (HTTPS)
- Implement proper access controls

### API Security
- Consider adding API keys
- Implement rate limiting
- Add request validation

## 📈 Scaling

### Auto-scaling
- Lambda: Automatic scaling based on demand
- DynamoDB: On-demand capacity
- API Gateway: Automatic scaling

### Performance Optimization
- Use Lambda provisioned concurrency for consistent performance
- Implement caching strategies
- Optimize database queries

## 🗑️ Cleanup

```bash
# Delete the CloudFormation stack
aws cloudformation delete-stack --stack-name spokhand-asl-lex-api-dev

# Wait for deletion to complete
aws cloudformation wait stack-delete-complete --stack-name spokhand-asl-lex-api-dev
```

## 📞 Support

For issues or questions:
1. Check CloudWatch logs
2. Review API Gateway metrics
3. Test endpoints individually
4. Contact the development team

## 🔗 Useful Links

- [API Gateway Console](https://console.aws.amazon.com/apigateway/)
- [Lambda Console](https://console.aws.amazon.com/lambda/)
- [DynamoDB Console](https://console.aws.amazon.com/dynamodb/)
- [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/) 