#!/bin/bash

# Get AWS account ID from existing configuration
AWS_ACCOUNT_ID=992382414589
AWS_REGION=${AWS_REGION:-us-east-1}

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names spokhand-upload-app || \
    aws ecr create-repository --repository-name spokhand-upload-app

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build and push Docker image
echo "Building and pushing Docker image..."
docker build -t spokhand-upload-app .
docker tag spokhand-upload-app:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/spokhand-upload-app:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/spokhand-upload-app:latest

# Create CloudWatch log group if it doesn't exist
aws logs describe-log-groups --log-group-name-prefix "/ecs/spokhand-upload-app" || \
    aws logs create-log-group --log-group-name "/ecs/spokhand-upload-app"

# Register task definition
echo "Registering task definition..."
envsubst < task-definition.json > task-definition-subst.json
aws ecs register-task-definition --cli-input-json file://task-definition-subst.json

# Create ECS cluster if it doesn't exist
aws ecs describe-clusters --clusters spokhand-cluster || \
    aws ecs create-cluster --cluster-name spokhand-cluster

# Create or update service
echo "Creating/updating ECS service..."
aws ecs describe-services --cluster spokhand-cluster --services spokhand-upload-service || \
    aws ecs create-service \
        --cluster spokhand-cluster \
        --service-name spokhand-upload-service \
        --task-definition spokhand-upload-app \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[subnet-0883c90b45adc8545],securityGroups=[sg-027e808c75e8d7e91],assignPublicIp=ENABLED}" \
        --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:${AWS_REGION}:${AWS_ACCOUNT_ID}:targetgroup/spokhand-tg/b10311c24bbcebae,containerName=spokhand-upload-app,containerPort=8501"

echo "Deployment completed. The application will be available at the load balancer URL." 