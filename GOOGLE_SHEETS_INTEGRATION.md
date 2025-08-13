# Google Sheets Integration for ASL-LEX Data Management

## Overview

The ASL-LEX Data Management system now includes Google Sheets integration, allowing data analysts to collaborate on ASL sign data collection using Google Sheets and automatically sync the data to the ASL-LEX database.

## Features

### 🔄 **Automatic Sync**
- Sync data from Google Sheets to ASL-LEX database
- Real-time status tracking
- Error handling and validation
- Support for large datasets

### 📊 **Data Validation**
- Automatic format validation
- Required field checking
- Data type conversion
- Duplicate detection

### 👥 **Collaborative Workflow**
- Multiple data analysts can work on the same sheet
- Version control and change tracking
- Conflict resolution
- Audit trail

## Setup Instructions

### 1. **Google Sheets API Setup**

#### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing project
3. Enable the Google Sheets API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

#### Step 2: Create API Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "API Key"
3. Copy the API key
4. (Optional) Restrict the API key to Google Sheets API only

#### Step 3: Configure Environment Variables
Add the API key to your environment:

```bash
# For development
export REACT_APP_GOOGLE_SHEETS_API_KEY="your_api_key_here"

# For production (add to your deployment environment)
REACT_APP_GOOGLE_SHEETS_API_KEY=your_api_key_here
```

### 2. **Google Sheets Template Setup**

#### Required Column Headers
Your Google Sheet must have the following columns (exact names):

| Column Name | Required | Type | Description |
|-------------|----------|------|-------------|
| Gloss | Yes | Text | Sign gloss (e.g., "HELLO") |
| English | Yes | Text | English translation |
| Handshape | No | Text | Handshape classification (A-Z) |
| Location | No | Text | Location on body |
| Movement | No | Text | Movement type |
| Palm Orientation | No | Text | Palm orientation |
| Dominant Hand | No | Text | Dominant hand shape |
| Non-Dominant Hand | No | Text | Non-dominant hand shape |
| Frequency | No | Number | Frequency in ASL-LEX corpus |
| Age of Acquisition | No | Number | Age of acquisition |
| Iconicity | No | Number | Iconicity rating (0-1) |
| Lexical Class | No | Text | Lexical class |
| Tags | No | Text | Comma-separated tags |
| Notes | No | Text | Additional notes |
| Video URL | No | URL | URL to sign video |
| Status | No | Text | pending/approved/rejected |
| Validation Status | No | Text | unvalidated/validated/needs_review |
| Confidence Score | No | Number | AI confidence score (0-1) |

#### Template Creation
1. Create a new Google Sheet
2. Add the required headers in the first row
3. Make the sheet publicly accessible (for API access)
4. Share the sheet URL with your team

### 3. **Frontend Configuration**

The ASL-LEX page is now accessible via:
- **Navigation**: Click "ASL-LEX" in the main navigation
- **Home Page**: Use the "Full ASL-LEX Manager" button
- **Direct URL**: `/ASLLex`

### 4. **Using the Integration**

#### Step 1: Connect Google Sheets
1. Navigate to the ASL-LEX page
2. Enter your Google Sheets URL in the "Google Sheets URL" field
3. Click "Open Sheet" to verify the URL
4. Click "Sync from Google Sheets" to start the sync process

#### Step 2: Monitor Sync Status
- The sync status will be displayed in real-time
- Check the "Last synced" timestamp
- Review any error messages if sync fails

#### Step 3: Validate Data
1. After sync, review the imported data in the ASL-LEX database
2. Use the validation tools to approve/reject signs
3. Add notes and confidence scores as needed

## API Endpoints

### Sync Endpoints
```
POST /api/asl-lex/google-sheets/sync
GET  /api/asl-lex/sync-status
```

### Request Format
```json
{
  "sheet_url": "https://docs.google.com/spreadsheets/d/...",
  "sheet_id": "extracted_sheet_id"
}
```

### Response Format
```json
{
  "success": true,
  "total": 150,
  "successful": 145,
  "failed": 5,
  "errors": ["Error details..."],
  "message": "Sync completed successfully"
}
```

## Data Flow

```
Google Sheets → Google Sheets API → ASL-LEX Backend → DynamoDB/S3
     ↓              ↓                    ↓              ↓
Data Entry → Authentication → Validation → Storage
```

## Error Handling

### Common Issues

1. **Invalid Google Sheets URL**
   - Ensure URL follows format: `https://docs.google.com/spreadsheets/d/SHEET_ID`
   - Check that the sheet is publicly accessible

2. **Missing API Key**
   - Verify `REACT_APP_GOOGLE_SHEETS_API_KEY` is set
   - Check API key restrictions

3. **Invalid Column Headers**
   - Ensure exact column names match the template
   - Check for extra spaces or typos

4. **Permission Issues**
   - Verify sheet sharing settings
   - Check API key permissions

### Troubleshooting

1. **Check Browser Console**
   - Look for JavaScript errors
   - Verify network requests

2. **Check Backend Logs**
   - Review AWS CloudWatch logs
   - Check API Gateway logs

3. **Test API Endpoints**
   - Use curl or Postman to test endpoints
   - Verify authentication

## Security Considerations

### API Key Security
- Never commit API keys to version control
- Use environment variables for configuration
- Restrict API keys to specific domains/IPs

### Data Privacy
- Ensure Google Sheets are properly shared
- Review data access permissions
- Implement audit logging

### Rate Limiting
- Google Sheets API has rate limits
- Implement exponential backoff
- Monitor API usage

## Future Enhancements

### Planned Features
1. **Real-time Sync**
   - Webhook integration
   - Automatic sync on sheet changes

2. **Advanced Validation**
   - AI-powered data validation
   - Duplicate detection
   - Quality scoring

3. **Collaboration Tools**
   - Multi-user editing
   - Comments and discussions
   - Version history

4. **Export Capabilities**
   - Export to Google Sheets
   - Multiple format support
   - Scheduled exports

## Support

For technical support:
1. Check this documentation
2. Review error logs
3. Contact the development team
4. Submit issues via GitHub

## Contributing

To contribute to the Google Sheets integration:
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit a pull request

---

**Note**: This integration is currently in development. Some features may not be fully implemented yet. Check the GitHub issues for current status and known limitations. 