#!/bin/bash

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if EB CLI is installed
if ! command -v eb &> /dev/null; then
    echo "EB CLI is not installed. Please install it first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "AWS_ACCESS_KEY_ID=your_access_key" > .env
    echo "AWS_SECRET_ACCESS_KEY=your_secret_key" >> .env
    echo "AWS_REGION=us-east-1" >> .env
    echo "S3_BUCKET_NAME=spokhand-data" >> .env
    echo "Please update the .env file with your AWS credentials"
    exit 1
fi

# Initialize EB application if not already initialized
if [ ! -d .elasticbeanstalk ]; then
    echo "Initializing Elastic Beanstalk application..."
    eb init -p python-3.9 spokhand-upload-app --region us-east-1
fi

# Create environment if it doesn't exist
if ! eb status &> /dev/null; then
    echo "Creating Elastic Beanstalk environment..."
    eb create spokhand-upload-env --instance-type t2.micro --single
fi

# Deploy the application
echo "Deploying application..."
eb deploy

# Get the application URL
echo "Getting application URL..."
APP_URL=$(eb status | grep CNAME | awk '{print $2}')
echo "Your application is available at: http://$APP_URL" 