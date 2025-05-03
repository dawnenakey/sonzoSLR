import boto3
import json
from botocore.exceptions import ClientError

def create_iam_user_and_policy():
    """Create IAM user and policy for Spokhand project"""
    # Initialize IAM client
    iam = boto3.client('iam')
    
    # User details
    username = 'spokhand-user'
    policy_name = 'SpokhandPolicy'
    
    try:
        # Create user
        print(f"Creating IAM user: {username}")
        iam.create_user(UserName=username)
        
        # Create access key
        print("Creating access key...")
        access_key_response = iam.create_access_key(UserName=username)
        access_key = access_key_response['AccessKey']
        
        # Create policy
        print(f"Creating policy: {policy_name}")
        with open('spokhand-policy.json', 'r') as policy_file:
            policy_document = policy_file.read()
        
        policy_response = iam.create_policy(
            PolicyName=policy_name,
            PolicyDocument=policy_document
        )
        
        # Attach policy to user
        print("Attaching policy to user...")
        iam.attach_user_policy(
            UserName=username,
            PolicyArn=policy_response['Policy']['Arn']
        )
        
        # Print credentials
        print("\n=== IMPORTANT: SAVE THESE CREDENTIALS ===")
        print(f"AWS Access Key ID: {access_key['AccessKeyId']}")
        print(f"AWS Secret Access Key: {access_key['SecretAccessKey']}")
        print("========================================")
        print("\nPlease save these credentials securely. You won't be able to see the Secret Access Key again.")
        
    except ClientError as e:
        print(f"Error setting up IAM: {str(e)}")
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print("User or policy already exists. Please choose a different name.")

if __name__ == '__main__':
    create_iam_user_and_policy() 