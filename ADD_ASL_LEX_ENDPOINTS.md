# Adding ASL-LEX Endpoints to Existing API Gateway

This guide shows you how to add the new ASL-LEX endpoints to your existing API Gateway and Lambda setup **without overwriting** your current functionality.

## 🚀 Quick Start

### Step 1: Add API Gateway Endpoints

```bash
# Make the script executable
chmod +x scripts/add-asl-lex-endpoints.sh

# Update the configuration in the script first, then run:
./scripts/add-asl-lex-endpoints.sh
```

**Important:** Update these values in the script to match your existing setup:
- `API_ID="qt8f7grhb5"` - Your existing API Gateway ID
- `STAGE_NAME="prod"` - Your existing stage name  
- `LAMBDA_FUNCTION_NAME="spokhand-processor"` - Your existing Lambda function name

### Step 2: Update Your Lambda Function

Replace your existing Lambda function code with the updated version:

```bash
# Copy the new Lambda handler
cp src/lambda_function_with_asl_lex.py src/lambda_function.py

# Update your Lambda function
aws lambda update-function-code \
  --function-name spokhand-processor \
  --zip-file fileb://lambda-deployment.zip
```

## 📋 What This Adds

### New API Endpoints
- `GET /api/asl-lex/signs` - List all signs with filtering
- `POST /api/asl-lex/signs` - Create a new sign
- `GET /api/asl-lex/sign-types` - Get available sign types
- `POST /api/asl-lex/sign-types` - Add custom sign type
- `GET /api/asl-lex/sign-types/custom` - Get custom sign types
- `POST /api/asl-lex/signs/batch-update-type` - Batch update sign types
- `POST /api/asl-lex/validate-asl-sign` - Validate ASL sign data
- `GET /api/asl-lex/analytics/sign-types` - Get sign type analytics
- `POST /api/asl-lex/bulk-upload` - Start bulk upload job
- `GET /api/asl-lex/bulk-upload/jobs` - List bulk upload jobs
- `GET /api/asl-lex/bulk-upload/template` - Download CSV template
- `GET /api/asl-lex/statistics` - Get system statistics

### Preserved Existing Endpoints
Your existing endpoints remain unchanged:
- `/sessions` - Session management
- `/videos/{videoId}/annotations` - Video annotations
- Any other existing endpoints

## 🔧 Configuration

### Update Your Frontend

Update your frontend API base URL to include the new endpoints:

```javascript
// In your frontend configuration
const API_BASE_URL = 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod';

// New ASL-LEX endpoints will be available at:
// ${API_BASE_URL}/api/asl-lex/signs
// ${API_BASE_URL}/api/asl-lex/sign-types
// etc.
```

### Environment Variables

Make sure your Lambda function has these environment variables:

```bash
DYNAMODB_TABLE=your-dynamodb-table-name
S3_BUCKET=your-s3-bucket-name
ENVIRONMENT=prod
```

## 🧪 Testing

### Test the New Endpoints

```bash
# Test sign types endpoint
curl -X GET "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/api/asl-lex/sign-types"

# Test analytics endpoint
curl -X GET "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/api/asl-lex/analytics/sign-types"

# Test validation endpoint
curl -X POST "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/api/asl-lex/validate-asl-sign" \
  -H "Content-Type: application/json" \
  -d '{
    "gloss": "HELLO",
    "handshape": "B",
    "location": "neutral space",
    "sign_type": "isolated_sign"
  }'
```

### Test Your Existing Endpoints

Your existing endpoints should continue to work:

```bash
# Test existing sessions endpoint
curl -X GET "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/sessions"

# Test existing annotations endpoint
curl -X GET "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/videos/123/annotations"
```

## 🔍 Verification

### Check API Gateway Resources

```bash
# List all resources in your API Gateway
aws apigateway get-resources --rest-api-id qt8f7grhb5
```

You should see the new `/api/asl-lex` resources added to your existing resources.

### Check Lambda Logs

```bash
# Monitor Lambda logs for the new endpoints
aws logs tail /aws/lambda/spokhand-processor --follow
```

## 🚨 Troubleshooting

### Common Issues

1. **API Gateway Resources Already Exist**
   - The script will fail if resources already exist
   - Check your API Gateway console to see what's already there
   - You may need to manually add missing resources

2. **Lambda Function Errors**
   - Check CloudWatch logs for detailed error messages
   - Verify all required environment variables are set
   - Ensure the ASL-LEX service dependencies are included

3. **CORS Issues**
   - The new endpoints include CORS headers
   - If you have specific CORS requirements, update the headers in the Lambda function

### Debug Commands

```bash
# Check API Gateway deployment status
aws apigateway get-deployments --rest-api-id qt8f7grhb5

# Test Lambda function directly
aws lambda invoke \
  --function-name spokhand-processor \
  --payload '{"httpMethod":"GET","path":"/api/asl-lex/sign-types"}' \
  response.json

# Check Lambda function configuration
aws lambda get-function --function-name spokhand-processor
```

## 📊 Monitoring

### CloudWatch Metrics
- Monitor API Gateway metrics for the new endpoints
- Check Lambda execution metrics
- Monitor DynamoDB read/write capacity

### Logs
- API Gateway logs: `/aws/apigateway/spokhand-api`
- Lambda logs: `/aws/lambda/spokhand-processor`

## 🔄 Rollback Plan

If you need to rollback:

1. **Remove API Gateway Resources**
   ```bash
   # Delete the /api/asl-lex resources manually from API Gateway console
   # Or use AWS CLI to delete specific resources
   ```

2. **Revert Lambda Function**
   ```bash
   # Restore your original Lambda function code
   aws lambda update-function-code \
     --function-name spokhand-processor \
     --zip-file fileb://original-lambda.zip
   ```

3. **Redeploy API Gateway**
   ```bash
   # Redeploy to remove the new endpoints
   aws apigateway create-deployment \
     --rest-api-id qt8f7grhb5 \
     --stage-name prod
   ```

## ✅ Success Checklist

- [ ] API Gateway endpoints added successfully
- [ ] Lambda function updated with new handlers
- [ ] All new endpoints responding correctly
- [ ] Existing endpoints still working
- [ ] Frontend updated to use new endpoints
- [ ] CORS working for all endpoints
- [ ] Error handling working properly
- [ ] Logs showing successful requests

## 📞 Support

If you encounter issues:

1. Check the CloudWatch logs for detailed error messages
2. Verify your API Gateway and Lambda configurations
3. Test endpoints individually to isolate problems
4. Review the troubleshooting section above

The new ASL-LEX endpoints are designed to work alongside your existing functionality without any conflicts! 