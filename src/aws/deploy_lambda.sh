#!/bin/bash

# Script to deploy the Spokhand Lambda function

FUNCTION_NAME="spokhand-processor"
ZIP_FILE="lambda_function.zip"
HANDLER="lambda_function.lambda_handler"
RUNTIME="python3.9"
# Make sure to use the correct role ARN from your AWS account
ROLE_ARN="arn:aws:iam::992382414589:role/spokhand-lambda-role"

echo "📦 Packaging Lambda function..."

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Remove old zip file if it exists
rm -f ${ZIP_FILE}

# Install dependencies to a temporary package directory
pip install --target ./package -r requirements.txt

# Create the zip file
cd package
zip -r ../${ZIP_FILE} .
cd ..

# Add Lambda function code to the zip
zip -g ${ZIP_FILE} lambda_function.py

echo "🚀 Deploying ${FUNCTION_NAME}..."

# Check if the function exists
aws lambda get-function --function-name ${FUNCTION_NAME} > /dev/null 2>&1

if [ $? -eq 0 ]; then
  # Function exists, update it
  echo "Function ${FUNCTION_NAME} exists. Updating function code..."
  aws lambda update-function-code \
    --function-name ${FUNCTION_NAME} \
    --zip-file fileb://${ZIP_FILE}
else
  # Function does not exist, create it
  echo "Function ${FUNCTION_NAME} does not exist. Creating new function..."
  aws lambda create-function \
    --function-name ${FUNCTION_NAME} \
    --runtime ${RUNTIME} \
    --role ${ROLE_ARN} \
    --handler ${HANDLER} \
    --zip-file fileb://${ZIP_FILE} \
    --timeout 30 \
    --memory-size 128
fi

# Clean up
rm -rf package
# Keep the zip file for inspection if needed
# rm -f ${ZIP_FILE}

echo "✅ Deployment complete!" 