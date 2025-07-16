# ASL-LEX Bulk Upload System

## Overview

The ASL-LEX Bulk Upload System allows data analysts to efficiently upload large datasets of ASL signs to the AWS S3 database. This system supports both CSV spreadsheets (metadata only) and ZIP files (videos + metadata) for comprehensive data management.

## Features

- **CSV Upload**: Upload metadata spreadsheets with sign information
- **ZIP Upload**: Upload videos and metadata together in compressed format
- **Progress Tracking**: Real-time progress monitoring for upload jobs
- **Error Handling**: Detailed error reporting for failed uploads
- **Template Download**: Pre-formatted CSV template for easy data entry
- **Job Management**: View, monitor, and cancel upload jobs
- **Validation**: Automatic data validation and error checking

## Getting Started

### 1. Access the Bulk Upload Interface

1. Navigate to the ASL-LEX Data Management page in your application
2. Click the "Bulk Upload" button to expand the upload interface
3. Download the CSV template for reference

### 2. Prepare Your Data

#### Option A: CSV Upload (Metadata Only)

1. **Download the Template**: Click "Download Template" to get the CSV template
2. **Fill in the Data**: Use the template to enter your sign metadata
3. **Upload the CSV**: Select your completed CSV file and upload

#### Option B: ZIP Upload (Videos + Metadata)

1. **Create a ZIP File**: Package your videos and CSV metadata together
2. **Organize Files**: Ensure your ZIP contains:
   - One CSV file with sign metadata
   - Video files referenced in the CSV
3. **Upload the ZIP**: Select your ZIP file and upload

## File Formats

### CSV Template Structure

The CSV template includes the following columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `gloss` | Yes | ASL gloss (sign name) | "HELLO" |
| `english` | Yes | English translation | "Hello" |
| `handshape` | No | Hand shape classification | "B" |
| `location` | No | Sign location on body | "neutral space" |
| `movement` | No | Movement description | "wave" |
| `palm_orientation` | No | Palm orientation | "palm forward" |
| `dominant_hand` | No | Dominant hand shape | "B" |
| `non_dominant_hand` | No | Non-dominant hand shape | "B" |
| `video_filename` | No* | Video file name (for ZIP uploads) | "hello.mp4" |
| `sign_type` | No | Type of sign video data | "isolated_sign" |
| `frequency` | No | Frequency score (0-100) | 95 |
| `age_of_acquisition` | No | Age of acquisition (years) | 2.5 |
| `iconicity` | No | Iconicity rating (0-1) | 0.8 |
| `lexical_class` | No | Lexical class | "interjection" |
| `tags` | No | Comma-separated tags | "greeting,common" |
| `notes` | No | Additional notes | "Common greeting sign" |
| `uploaded_by` | No | Uploader identifier | "data_analyst" |

*Required for ZIP uploads to match videos with metadata

### Sign Types

The system supports various sign types for classification:

| Sign Type | Description | Use Case |
|-----------|-------------|----------|
| `isolated_sign` | Single sign performed in isolation | Basic vocabulary training |
| `continuous_signing` | Sign within continuous signing context | Natural language processing |
| `fingerspelling` | Manual alphabet or fingerspelling | Alphabet and spelling training |
| `classifier` | Classifier construction or classifier predicate | Spatial grammar training |
| `compound_sign` | Compound sign combining multiple elements | Complex sign recognition |
| `inflected_sign` | Sign with grammatical inflection | Grammar training |
| `directional_sign` | Sign with directional movement | Directional verb training |
| `spatial_sign` | Sign involving spatial relationships | Spatial concept training |
| `temporal_sign` | Sign indicating time or temporal aspect | Temporal concept training |
| `manner_sign` | Sign indicating manner of action | Action description training |
| `number_sign` | Number or numeral sign | Number recognition |
| `question_sign` | Question or interrogative sign | Question formation training |
| `negation_sign` | Negative or negation sign | Negation training |
| `modality_sign` | Sign indicating modality or mood | Modal expression training |
| `other` | Other type of sign not listed above | Miscellaneous training |

### Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- WebM

### ZIP File Structure

```
your_upload.zip
├── metadata.csv          # CSV file with sign metadata
├── hello.mp4            # Video file referenced in CSV
├── thank_you.mp4        # Video file referenced in CSV
└── good.mp4             # Video file referenced in CSV
```

## Upload Process

### Individual Video Upload

For uploading single videos with metadata:

1. **Access Individual Upload**: Click "Individual Upload" in the interface
2. **Select Video**: Choose your video file (MP4, AVI, MOV, WebM)
3. **Enter Basic Information**:
   - **Gloss**: ASL sign name (e.g., "HELLO")
   - **English**: English translation (e.g., "Hello")
   - **Sign Type**: Select the type of sign video data
   - **Uploaded By**: Your name or identifier
4. **Add Advanced Metadata** (optional):
   - Handshape, location, movement
   - Lexical class, tags, notes
5. **Upload**: Click "Upload Video with Metadata"

### Bulk Upload

#### Step-by-Step Instructions

1. **Prepare Your Files**
   - For CSV: Ensure all required fields are filled
   - For ZIP: Verify video filenames match CSV references

2. **Upload Process**
   - Click "Bulk Upload" to expand the interface
   - Select your file (CSV or ZIP)
   - Enter your name/ID in "Uploaded By" field
   - Click "Upload and Process"

3. **Monitor Progress**
   - Watch the progress bar during upload
   - Check the "Bulk Upload Jobs" section for status
   - Review any error messages if upload fails

4. **Review Results**
   - Check job status (completed, completed_with_errors, failed)
   - Review error logs for any issues
   - Verify uploaded signs in the main interface

## Error Handling

### Common Issues and Solutions

#### CSV Upload Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Missing required field" | Required columns missing | Ensure gloss and english are filled |
| "Invalid data type" | Wrong data format | Check number fields (frequency, age_of_acquisition, iconicity) |
| "Invalid handshape" | Unknown handshape value | Use standard handshape codes (A-Z) |

#### ZIP Upload Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "No CSV file found" | Missing metadata file | Include exactly one CSV file in ZIP |
| "Video file not found" | Missing video file | Ensure video_filename matches actual files |
| "Invalid video format" | Unsupported video type | Convert to MP4, AVI, MOV, or WebM |

#### General Upload Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "File too large" | ZIP/CSV exceeds size limit | Split into smaller batches |
| "Network timeout" | Slow connection | Try again or use smaller files |
| "AWS S3 error" | Storage service issue | Contact administrator |

## Best Practices

### Data Preparation

1. **Use the Template**: Always start with the provided CSV template
2. **Validate Data**: Check for typos and missing required fields
3. **Consistent Formatting**: Use standard values for handshapes, locations, etc.
4. **Test Small Batches**: Upload a few signs first to verify format

### File Organization

1. **Descriptive Names**: Use clear, descriptive filenames
2. **Consistent Structure**: Follow the same organization pattern
3. **Backup Data**: Keep original files as backup
4. **Version Control**: Use version numbers for large datasets

### Upload Strategy

1. **Batch Size**: Upload 100-500 signs per batch for optimal performance
2. **Network Stability**: Ensure stable internet connection
3. **Monitor Resources**: Check system resources during large uploads
4. **Error Review**: Always review error logs after upload

## Job Management

### Viewing Upload Jobs

1. Navigate to "Bulk Upload Jobs" section
2. View all recent upload jobs with status indicators
3. Click the eye icon to see detailed job information

### Job Statuses

- **Processing**: Upload is being processed
- **Completed**: All items uploaded successfully
- **Completed with Errors**: Some items failed, others succeeded
- **Failed**: Upload failed completely
- **Cancelled**: Upload was cancelled by user

### Job Details

Each job shows:
- **File Information**: Filename, type, upload date
- **Progress**: Items processed vs. total
- **Results**: Successful and failed item counts
- **Error Log**: Detailed error messages for failed items

### Cancelling Jobs

- Only processing jobs can be cancelled
- Click the X icon next to a processing job
- Cancelled jobs cannot be resumed

## API Endpoints

### Upload Endpoints

```
POST /api/asl-lex/bulk-upload
- Upload CSV or ZIP files
- Returns job ID and processing status

GET /api/asl-lex/bulk-upload/template
- Download CSV template
- Returns formatted CSV file
```

### Job Management Endpoints

```
GET /api/asl-lex/bulk-upload/jobs
- List all upload jobs
- Optional status filter

GET /api/asl-lex/bulk-upload/jobs/{job_id}
- Get specific job details
- Returns progress and error information

POST /api/asl-lex/bulk-upload/jobs/{job_id}/cancel
- Cancel a processing job
- Cannot cancel completed/failed jobs
```

## Data Validation

### Automatic Validation

The system automatically validates:
- Required field presence
- Data type correctness
- File format compatibility
- Video file existence (for ZIP uploads)

### Validation Rules

| Field | Validation Rule |
|-------|----------------|
| `gloss` | Required, non-empty string |
| `english` | Required, non-empty string |
| `handshape` | Must be A-Z or empty |
| `frequency` | Must be 0-100 or empty |
| `age_of_acquisition` | Must be positive number or empty |
| `iconicity` | Must be 0-1 or empty |
| `video_filename` | Must match actual file (ZIP uploads) |

## Troubleshooting

### Upload Issues

**Problem**: Upload fails immediately
- **Solution**: Check file size and format
- **Solution**: Verify network connection
- **Solution**: Ensure file is not corrupted

**Problem**: Some items fail to upload
- **Solution**: Check error logs for specific issues
- **Solution**: Fix data format issues
- **Solution**: Re-upload corrected data

**Problem**: Progress stops at certain percentage
- **Solution**: Wait for processing to complete
- **Solution**: Check for large video files
- **Solution**: Contact administrator if stuck

### Data Issues

**Problem**: Signs not appearing in database
- **Solution**: Check job status for completion
- **Solution**: Verify data format matches template
- **Solution**: Review error logs for validation failures

**Problem**: Videos not linked to signs
- **Solution**: Ensure video_filename matches actual files
- **Solution**: Check video format compatibility
- **Solution**: Verify ZIP file structure

## Support

### Getting Help

1. **Check Error Logs**: Review detailed error messages
2. **Use Template**: Always use the provided CSV template
3. **Test Small Batches**: Upload a few items first
4. **Contact Administrator**: For persistent issues

### Contact Information

- **Technical Support**: Contact your system administrator
- **Data Questions**: Consult ASL-LEX documentation
- **Format Issues**: Review this documentation

## Advanced Features

### Batch Processing

- Upload multiple files in sequence
- Monitor all jobs simultaneously
- Cancel individual jobs as needed

### Data Export

- Export uploaded data in various formats
- Generate reports on upload statistics
- Download processed data for analysis

### Integration

- API access for automated uploads
- Webhook notifications for job completion
- Integration with external data sources

## Security Considerations

### Data Protection

- All uploads are encrypted in transit
- Data is stored securely in AWS S3
- Access is controlled through authentication
- Audit logs track all upload activities

### Best Practices

- Use secure connections for uploads
- Don't share sensitive data in file names
- Regularly review uploaded data
- Report any security concerns immediately

---

*This bulk upload system is designed to streamline the process of adding ASL-LEX data to your sign language recognition training database. For questions or issues, please refer to the troubleshooting section or contact your system administrator.* 