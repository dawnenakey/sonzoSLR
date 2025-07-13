# Dataset Upload Guide for AWS Integration

This guide explains how to upload your WLASL videos (.webm files) and ASL-LEX CSV files to AWS and view them in your Amplify platform.

## Prerequisites

1. **AWS Credentials**: Make sure you have AWS credentials configured
2. **Python Dependencies**: Install required packages
3. **Local Files**: Have your WLASL videos and ASL-LEX CSV files ready

## Setup

### 1. Install Dependencies

```bash
# Install required Python packages
pip install boto3 pandas

# Or if using requirements.txt
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Set up your AWS credentials using one of these methods:

**Option A: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

**Option B: AWS CLI Configuration**
```bash
aws configure
```

**Option C: Create .env file**
```bash
# Create .env file in project root
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET=spokhand-data
DYNAMODB_TABLE=spokhand-data-collection
```

## Uploading WLASL Videos

### Step 1: Prepare Your Video Files

Organize your WLASL videos in a directory structure like:
```
wlasl_videos/
├── train/
│   ├── hello_001.webm
│   ├── goodbye_002.webm
│   └── ...
├── val/
│   ├── hello_003.webm
│   └── ...
└── test/
    ├── hello_004.webm
    └── ...
```

### Step 2: Upload Videos to AWS

**Basic Upload:**
```bash
python scripts/upload_wlasl_videos.py /path/to/your/wlasl_videos
```

**With WLASL Metadata JSON:**
```bash
python scripts/upload_wlasl_videos.py /path/to/your/wlasl_videos \
  --wlasl-json /path/to/WLASL_v0.3.json
```

**Dry Run (Preview Only):**
```bash
python scripts/upload_wlasl_videos.py /path/to/your/wlasl_videos --dry-run
```

**Custom File Extensions:**
```bash
python scripts/upload_wlasl_videos.py /path/to/your/wlasl_videos \
  --extensions .webm .mp4 .avi
```

**Batch Upload with Progress:**
```bash
python scripts/upload_wlasl_videos.py /path/to/your/wlasl_videos \
  --batch-size 20
```

### Step 3: Verify Upload

The script will show:
- Number of videos found
- Upload progress
- Success/failure summary
- Video metadata extracted

## Uploading ASL-LEX CSV Files

### Step 1: Prepare Your CSV Files

Place your ASL-LEX CSV files in a directory:
```
asl_lex_data/
├── signdata.csv
├── handshape_data.csv
├── location_data.csv
└── ...
```

### Step 2: Upload CSV Files to AWS

**Basic Upload:**
```bash
python scripts/upload_asl_lex_csv.py /path/to/your/asl_lex_data
```

**Preview Data Before Upload:**
```bash
python scripts/upload_asl_lex_csv.py /path/to/your/asl_lex_data --preview
```

**Dry Run:**
```bash
python scripts/upload_asl_lex_csv.py /path/to/your/asl_lex_data --dry-run
```

### Step 3: Verify Upload

The script will show:
- CSV files found
- Row and column counts
- Data type detection
- Upload results

## Viewing Data in Amplify Platform

### Step 1: Access the Dataset Viewer

1. Navigate to your Amplify app
2. Go to the Dataset Viewer component
3. You'll see two tabs: "WLASL Videos" and "ASL-LEX CSV"

### Step 2: Browse WLASL Videos

**Features:**
- Search videos by gloss or filename
- Filter by split (train/val/test)
- View video metadata
- Play videos directly
- Download video files

**Usage:**
1. Use the search bar to find specific signs
2. Select a split to filter videos
3. Click on a video card to see details
4. Click "Play" to stream the video
5. Use "View in S3" to access the original file

### Step 3: Browse ASL-LEX CSV Data

**Features:**
- View CSV file metadata
- Preview data structure
- Download CSV files
- Search within data
- View sample data

**Usage:**
1. Click on a CSV file card to view details
2. See column names and data types
3. Preview sample data
4. Download the full CSV file
5. Use "View in S3" to access the original file

## API Endpoints

The following API endpoints are available for programmatic access:

### WLASL Videos
- `GET /api/wlasl-videos` - List all videos
- `GET /api/wlasl-videos?split=train` - Filter by split
- `GET /api/wlasl-videos?gloss=hello` - Filter by gloss
- `POST /api/wlasl-videos/{id}/stream-url` - Generate streaming URL

### ASL-LEX CSV Files
- `GET /api/asl-lex-files` - List all CSV files
- `GET /api/asl-lex-data/{file_id}` - Get CSV data
- `POST /api/asl-lex-files/{file_id}/download` - Generate download URL

## Data Storage Structure

### S3 Bucket Organization
```
spokhand-data/
├── wlasl_videos/
│   ├── train/
│   │   ├── video_id_1.webm
│   │   └── video_id_2.webm
│   ├── val/
│   └── test/
└── asl_lex_data/
    ├── file_id_1.csv
    └── file_id_2.csv
```

### DynamoDB Schema

**WLASL Videos:**
```json
{
  "video_id": "uuid",
  "s3_key": "wlasl_videos/train/video_id.webm",
  "bucket": "spokhand-data",
  "presigned_url": "https://...",
  "file_format": "webm",
  "dataset": "WLASL",
  "gloss": "hello",
  "split": "train",
  "instance_id": "001",
  "created_at": "2024-01-01T00:00:00Z",
  "status": "uploaded"
}
```

**ASL-LEX CSV Files:**
```json
{
  "file_id": "uuid",
  "s3_key": "asl_lex_data/file_id.csv",
  "bucket": "spokhand-data",
  "presigned_url": "https://...",
  "file_format": "csv",
  "dataset": "ASL-LEX",
  "filename": "signdata.csv",
  "description": "ASL-LEX dataset file",
  "row_count": 1000,
  "column_count": 15,
  "columns": ["gloss", "handshape", "location", ...],
  "data_type": "gloss_data",
  "created_at": "2024-01-01T00:00:00Z",
  "status": "uploaded"
}
```

## Troubleshooting

### Common Issues

**1. AWS Credentials Error**
```
Error: No credentials found
```
**Solution:** Configure AWS credentials using one of the methods above.

**2. S3 Upload Permission Error**
```
Error: Access Denied
```
**Solution:** Ensure your AWS user has S3 and DynamoDB permissions.

**3. File Not Found**
```
Error: Directory does not exist
```
**Solution:** Check the path to your video/CSV files.

**4. Large File Upload Timeout**
```
Error: Upload timeout
```
**Solution:** Use smaller batch sizes or increase timeout settings.

### Performance Tips

1. **Batch Uploads**: Use `--batch-size` to control upload batches
2. **Parallel Processing**: The scripts handle multiple files efficiently
3. **Progress Monitoring**: Watch the console output for progress
4. **Dry Runs**: Always test with `--dry-run` first

### Monitoring Uploads

Check your AWS Console:
1. **S3 Console**: Verify files are uploaded
2. **DynamoDB Console**: Check metadata is stored
3. **CloudWatch**: Monitor Lambda function logs

## Security Considerations

1. **Presigned URLs**: Automatically expire after 1 hour
2. **Access Control**: Use IAM roles and policies
3. **Data Encryption**: S3 objects are encrypted at rest
4. **Network Security**: Use HTTPS for all transfers

## Cost Optimization

1. **S3 Storage**: Videos are stored in S3 (pay per GB)
2. **DynamoDB**: Metadata stored in DynamoDB (pay per request)
3. **Data Transfer**: Consider CloudFront for video streaming
4. **Lifecycle Policies**: Configure S3 lifecycle rules for old data

## Next Steps

After uploading your data:

1. **Train Models**: Use the uploaded data for ML training
2. **Build Applications**: Create applications using the data
3. **Share Data**: Generate shareable links for collaborators
4. **Analyze Usage**: Monitor data access patterns

## Support

For issues with:
- **Upload Scripts**: Check the console output for error messages
- **AWS Permissions**: Verify IAM roles and policies
- **Amplify Integration**: Check browser console for API errors
- **Data Access**: Ensure presigned URLs haven't expired 