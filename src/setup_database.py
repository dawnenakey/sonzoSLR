"""
Database setup script for SPOKHAND SIGNCUT
Creates DynamoDB tables for users, audit logs, and Epic 2 text corpus management
"""

import boto3
import json
from botocore.exceptions import ClientError

def create_users_table():
    """Create the users table"""
    dynamodb = boto3.resource('dynamodb')
    
    table_name = 'spokhand-users'
    
    try:
        # Check if table already exists
        existing_table = dynamodb.Table(table_name)
        existing_table.load()
        print(f"Table {table_name} already exists")
        return existing_table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            pass
        else:
            raise
    
    # Create table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'id',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'id',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'email',
                'AttributeType': 'S'  # String
            }
        ],
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'EmailIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'email',
                        'KeyType': 'HASH'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    
    # Wait for table to be created
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
    print(f"Created table {table_name}")
    return table

def create_audit_logs_table():
    """Create the audit logs table"""
    dynamodb = boto3.resource('dynamodb')
    
    table_name = 'spokhand-audit-logs'
    
    try:
        # Check if table already exists
        existing_table = dynamodb.Table(table_name)
        existing_table.load()
        print(f"Table {table_name} already exists")
        return existing_table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            pass
        else:
            raise
    
    # Create table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'id',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'id',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'timestamp',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'user_id',
                'AttributeType': 'S'  # String
            }
        ],
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'TimestampIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'timestamp',
                        'KeyType': 'HASH'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            {
                'IndexName': 'UserIdIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'user_id',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'timestamp',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    
    # Wait for table to be created
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
    print(f"Created table {table_name}")
    return table

def create_text_corpora_table():
    """Create the text corpora table for Epic 2"""
    dynamodb = boto3.resource('dynamodb')
    
    table_name = 'spokhand-text-corpora'
    
    try:
        # Check if table already exists
        existing_table = dynamodb.Table(table_name)
        existing_table.load()
        print(f"Table {table_name} already exists")
        return existing_table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            pass
        else:
            raise
    
    # Create table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'id',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'id',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'created_by',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'status',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'language',
                'AttributeType': 'S'  # String
            }
        ],
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'CreatedByIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'created_by',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'id',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            {
                'IndexName': 'StatusIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'status',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'created_at',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            {
                'IndexName': 'LanguageIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'language',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'created_at',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    
    # Wait for table to be created
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
    print(f"Created table {table_name}")
    return table

def create_text_segments_table():
    """Create the text segments table for Epic 2"""
    dynamodb = boto3.resource('dynamodb')
    
    table_name = 'spokhand-text-segments'
    
    try:
        # Check if table already exists
        existing_table = dynamodb.Table(table_name)
        existing_table.load()
        print(f"Table {table_name} already exists")
        return existing_table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            pass
        else:
            raise
    
    # Create table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'id',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'id',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'corpus_id',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'status',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'segment_type',
                'AttributeType': 'S'  # String
            }
        ],
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'CorpusIdIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'corpus_id',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'position',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            {
                'IndexName': 'StatusIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'status',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'corpus_id',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            {
                'IndexName': 'SegmentTypeIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'segment_type',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'corpus_id',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    
    # Wait for table to be created
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
    print(f"Created table {table_name}")
    return table

def create_corpus_exports_table():
    """Create the corpus exports table for Epic 2"""
    dynamodb = boto3.resource('dynamodb')
    
    table_name = 'spokhand-corpus-exports'
    
    try:
        # Check if table already exists
        existing_table = dynamodb.Table(table_name)
        existing_table.load()
        print(f"Table {table_name} already exists")
        return existing_table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            pass
        else:
            raise
    
    # Create table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'id',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'id',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'corpus_id',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'status',
                'AttributeType': 'S'  # String
            },
            {
                'AttributeName': 'created_by',
                'AttributeType': 'S'  # String
            }
        ],
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'CorpusIdIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'corpus_id',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'created_at',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            {
                'IndexName': 'StatusIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'status',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'created_at',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            {
                'IndexName': 'CreatedByIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'created_by',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'created_at',
                        'KeyType': 'RANGE'
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    
    # Wait for table to be created
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
    print(f"Created table {table_name}")
    return table

def create_sample_text_corpus():
    """Create a sample text corpus for demonstration"""
    try:
        from text_corpus_service import TextCorpusService
        
        text_service = TextCorpusService()
        
        # Create a sample ASL corpus
        sample_corpus = text_service.create_corpus(
            name="Basic ASL Vocabulary",
            description="A collection of basic ASL signs and phrases for beginners",
            language="ASL",
            created_by="admin@spokhand.com",
            metadata={
                "difficulty_level": "beginner",
                "target_audience": "students",
                "curriculum_type": "vocabulary"
            },
            tags=["beginner", "vocabulary", "education"]
        )
        
        # Add some sample text segments
        sample_segments = [
            {
                "text": "Hello, how are you?",
                "segment_type": "phrase",
                "metadata": {"category": "greetings", "difficulty": 1}
            },
            {
                "text": "My name is...",
                "segment_type": "phrase", 
                "metadata": {"category": "introductions", "difficulty": 1}
            },
            {
                "text": "Thank you very much",
                "segment_type": "phrase",
                "metadata": {"category": "courtesy", "difficulty": 1}
            },
            {
                "text": "I don't understand",
                "segment_type": "phrase",
                "metadata": {"category": "communication", "difficulty": 2}
            },
            {
                "text": "Please repeat that",
                "segment_type": "phrase",
                "metadata": {"category": "communication", "difficulty": 2}
            }
        ]
        
        for segment_data in sample_segments:
            text_service.add_text_segment(
                corpus_id=sample_corpus.id,
                text=segment_data["text"],
                segment_type=segment_data["segment_type"],
                created_by="admin@spokhand.com",
                metadata=segment_data["metadata"]
            )
        
        print(f"Created sample corpus: {sample_corpus.name} with {len(sample_segments)} segments")
        return sample_corpus
        
    except Exception as e:
        print(f"Error creating sample corpus: {e}")
        return None

def create_video_text_links_table():
    """Create the video-text links table for Epic 3."""
    try:
        table_name = f"{TABLE_PREFIX}-video-text-links"
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'id', 'KeyType': 'HASH'}  # Partition key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'id', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'VideoIndex',
                    'KeySchema': [
                        {'AttributeName': 'video_id', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'video_id', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'CorpusIndex',
                    'KeySchema': [
                        {'AttributeName': 'corpus_id', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'corpus_id', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'LinkTypeIndex',
                    'KeySchema': [
                        {'AttributeName': 'link_type', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'link_type', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        
        # Wait for table to be created
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        print(f"Created table: {table_name}")
        return table
        
    except Exception as e:
        print(f"Error creating video-text links table: {e}")
        return None

def create_video_text_annotations_table():
    """Create the video-text annotations table for Epic 3."""
    try:
        table_name = f"{TABLE_PREFIX}-video-text-annotations"
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'id', 'KeyType': 'HASH'}  # Partition key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'id', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'VideoIndex',
                    'KeySchema': [
                        {'AttributeName': 'video_id', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'video_id', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'AnnotationTypeIndex',
                    'KeySchema': [
                        {'AttributeName': 'annotation_type', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'annotation_type', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'StatusIndex',
                    'KeySchema': [
                        {'AttributeName': 'status', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'status', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        
        # Wait for table to be created
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        print(f"Created table: {table_name}")
        return table
        
    except Exception as e:
        print(f"Error creating video-text annotations table: {e}")
        return None

def create_video_text_exports_table():
    """Create the video-text exports table for Epic 3."""
    try:
        table_name = f"{TABLE_PREFIX}-video-text-exports"
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'id', 'KeyType': 'HASH'}  # Partition key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'id', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'VideoIndex',
                    'KeySchema': [
                        {'AttributeName': 'video_id', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'video_id', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'StatusIndex',
                    'KeySchema': [
                        {'AttributeName': 'status', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'status', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'CreatedByIndex',
                    'KeySchema': [
                        {'AttributeName': 'created_by', 'KeyType': 'HASH'}
                    ],
                    'AttributeDefinitions': [
                        {'AttributeName': 'created_by', 'AttributeType': 'S'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        
        # Wait for table to be created
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        print(f"Created table: {table_name}")
        return table
        
    except Exception as e:
        print(f"Error creating video-text exports table: {e}")
        return None

def create_sample_video_text_data():
    """Create sample video-text data for demonstration purposes."""
    try:
        # Initialize video-text linking service
        from video_text_linking_service import VideoTextLinkingService
        video_text_service = VideoTextLinkingService()
        
        # Create sample video-text annotation
        sample_annotation = video_text_service.create_video_text_annotation(
            video_id="sample-video-001",
            start_time=0.0,
            end_time=3.5,
            annotation_type="sign_unit",
            text_content="Hello, how are you?",
            created_by="admin@spokhand.com",
            linked_corpus_id="sample-corpus-001",  # This will be created by Epic 2
            linked_text_segment_id="sample-segment-001",
            confidence_score=0.95,
            tags=["greeting", "beginner", "sample"]
        )
        
        print(f"Created sample video-text annotation: {sample_annotation.id}")
        return sample_annotation
        
    except Exception as e:
        print(f"Error creating sample video-text data: {e}")
        return None

def main():
    """Main function to create all tables"""
    print("Setting up SPOKHAND SIGNCUT database...")
    print("=" * 50)
    
    try:
        # Create Epic 1 tables
        print("\n1. Creating Epic 1 tables...")
        users_table = create_users_table()
        audit_logs_table = create_audit_logs_table()
        
        # Create Epic 2 tables
        print("\n2. Creating Epic 2 tables...")
        text_corpora_table = create_text_corpora_table()
        text_segments_table = create_text_segments_table()
        corpus_exports_table = create_corpus_exports_table()
        
        # Create Epic 3 tables
        print("\n3. Creating Epic 3 tables...")
        video_text_links_table = create_video_text_links_table()
        video_text_annotations_table = create_video_text_annotations_table()
        video_text_exports_table = create_video_text_exports_table()
        
        print("\n4. Creating sample data...")
        sample_corpus = create_sample_text_corpus()
        sample_video_text = create_sample_video_text_data()
        
        print("\n" + "=" * 50)
        print("Database setup completed successfully!")
        print("\nTables created:")
        print(f"  ✓ {users_table.name}")
        print(f"  ✓ {audit_logs_table.name}")
        print(f"  ✓ {text_corpora_table.name}")
        print(f"  ✓ {text_segments_table.name}")
        print(f"  ✓ {corpus_exports_table.name}")
        print(f"  ✓ {video_text_links_table.name}")
        print(f"  ✓ {video_text_annotations_table.name}")
        print(f"  ✓ {video_text_exports_table.name}")
        
        if sample_corpus:
            print(f"\nSample data created:")
            print(f"  ✓ Sample corpus: {sample_corpus.name}")
        
        if sample_video_text:
            print(f"  ✓ Sample video-text annotation: {sample_video_text.id}")
        
        print("\nEpic 3 (Enhanced Video Workspace) is ready!")
        print("You can now start the unified video-text API with: python video_text_api.py")
        
    except Exception as e:
        print(f"\nError during database setup: {e}")
        print("Please check your AWS credentials and permissions")
        raise

if __name__ == "__main__":
    main() 