import boto3
import os
import zipfile
import shutil
from botocore.exceptions import ClientError

def create_lambda_package():
    """Create a zip package for the Lambda function"""
    # Create a temporary directory
    if os.path.exists('lambda_package'):
        shutil.rmtree('lambda_package')
    os.makedirs('lambda_package')
    
    # Copy the Lambda function
    shutil.copy('lambda_function.py', 'lambda_package/')
    
    # Create requirements.txt for Lambda
    with open('lambda_package/requirements.txt', 'w') as f:
        f.write('boto3==1.34.0\n')
    
    # Create the zip file
    with zipfile.ZipFile('lambda_function.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('lambda_package'):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, 'lambda_package')
                zipf.write(file_path, arcname)
    
    # Clean up
    shutil.rmtree('lambda_package')

def deploy_infrastructure():
    """Deploy the AWS infrastructure using CloudFormation"""
    # Initialize AWS clients
    cf_client = boto3.client('cloudformation')
    s3_client = boto3.client('s3')
    
    # Create the Lambda package
    create_lambda_package()
    
    try:
        # Upload Lambda package to S3
        s3_client.upload_file(
            'lambda_function.zip',
            'spokhand-data',
            'lambda/lambda_function.zip'
        )
        
        # Deploy CloudFormation stack
        with open('infrastructure.yaml', 'r') as template_file:
            template_body = template_file.read()
        
        response = cf_client.create_stack(
            StackName='spokhand-stack',
            TemplateBody=template_body,
            Capabilities=['CAPABILITY_IAM']
        )
        
        print(f"Stack creation initiated: {response['StackId']}")
        print("Waiting for stack creation to complete...")
        
        # Wait for stack creation to complete
        waiter = cf_client.get_waiter('stack_create_complete')
        waiter.wait(StackName='spokhand-stack')
        
        print("Stack creation completed successfully!")
        
        # Get stack outputs
        stack = cf_client.describe_stacks(StackName='spokhand-stack')['Stacks'][0]
        print("\nStack Outputs:")
        for output in stack.get('Outputs', []):
            print(f"{output['OutputKey']}: {output['OutputValue']}")
            
    except ClientError as e:
        print(f"Error deploying infrastructure: {str(e)}")
    finally:
        # Clean up
        if os.path.exists('lambda_function.zip'):
            os.remove('lambda_function.zip')

if __name__ == '__main__':
    deploy_infrastructure() 