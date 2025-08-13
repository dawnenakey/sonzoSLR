// Google Sheets API Client for ASL-LEX Data Management
// This service handles syncing data between Google Sheets and the ASL-LEX database

const GOOGLE_SHEETS_API_BASE = 'https://sheets.googleapis.com/v4/spreadsheets';

class GoogleSheetsClient {
  constructor() {
    this.apiKey = process.env.REACT_APP_GOOGLE_SHEETS_API_KEY;
    this.baseUrl = process.env.REACT_APP_API_URL || 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod';
  }

  // Extract sheet ID from Google Sheets URL
  extractSheetId(url) {
    const match = url.match(/\/spreadsheets\/d\/([a-zA-Z0-9-_]+)/);
    return match ? match[1] : null;
  }

  // Get sheet data from Google Sheets API
  async getSheetData(sheetId, range = 'A1:Z1000') {
    if (!this.apiKey) {
      throw new Error('Google Sheets API key not configured');
    }

    const url = `${GOOGLE_SHEETS_API_BASE}/${sheetId}/values/${range}?key=${this.apiKey}`;
    
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch sheet data: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.values || [];
    } catch (error) {
      console.error('Error fetching Google Sheets data:', error);
      throw error;
    }
  }

  // Convert Google Sheets data to ASL-LEX format
  convertToASLLexFormat(sheetData) {
    if (!sheetData || sheetData.length < 2) {
      throw new Error('Sheet data is empty or missing headers');
    }

    const headers = sheetData[0];
    const rows = sheetData.slice(1);
    
    // Expected column mapping
    const columnMap = {
      'Gloss': 'gloss',
      'English': 'english',
      'Handshape': 'handshape',
      'Location': 'location',
      'Movement': 'movement',
      'Palm Orientation': 'palm_orientation',
      'Dominant Hand': 'dominant_hand',
      'Non-Dominant Hand': 'non_dominant_hand',
      'Frequency': 'frequency',
      'Age of Acquisition': 'age_of_acquisition',
      'Iconicity': 'iconicity',
      'Lexical Class': 'lexical_class',
      'Tags': 'tags',
      'Notes': 'notes',
      'Video URL': 'video_url',
      'Status': 'status',
      'Validation Status': 'validation_status',
      'Confidence Score': 'confidence_score'
    };

    // Create header index mapping
    const headerIndex = {};
    headers.forEach((header, index) => {
      const normalizedHeader = header.trim();
      if (columnMap[normalizedHeader]) {
        headerIndex[columnMap[normalizedHeader]] = index;
      }
    });

    // Convert rows to ASL-LEX format
    const aslLexData = rows.map((row, rowIndex) => {
      const signData = {
        id: `sheet_${rowIndex + 1}`,
        uploaded_at: new Date().toISOString(),
        uploaded_by: 'google_sheets_sync'
      };

      // Map each column
      Object.entries(headerIndex).forEach(([field, index]) => {
        const value = row[index] || '';
        
        // Convert data types appropriately
        switch (field) {
          case 'frequency':
          case 'age_of_acquisition':
          case 'iconicity':
          case 'confidence_score':
            signData[field] = parseFloat(value) || 0;
            break;
          case 'tags':
            signData[field] = value ? value.split(',').map(tag => tag.trim()) : [];
            break;
          default:
            signData[field] = value;
        }
      });

      return signData;
    });

    return aslLexData;
  }

  // Sync data from Google Sheets to ASL-LEX database
  async syncFromGoogleSheets(sheetUrl) {
    try {
      const sheetId = this.extractSheetId(sheetUrl);
      if (!sheetId) {
        throw new Error('Invalid Google Sheets URL');
      }

      // Use the backend sync endpoint
      const response = await fetch(`${this.baseUrl}/api/asl-lex/google-sheets/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sheet_url: sheetUrl,
          sheet_id: sheetId
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      return {
        total: result.total || 0,
        successful: result.successful || 0,
        failed: result.failed || 0,
        errors: result.errors || [],
        message: result.message || 'Sync completed'
      };
    } catch (error) {
      console.error('Error syncing from Google Sheets:', error);
      throw error;
    }
  }

  // Upload individual sign data to ASL-LEX database
  async uploadToASLLexDatabase(signData) {
    try {
      const response = await fetch(`${this.baseUrl}/api/asl-lex/signs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(signData)
      });

      if (!response.ok) {
        throw new Error(`Failed to upload sign: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error uploading to ASL-LEX database:', error);
      throw error;
    }
  }

  // Get sync status and history
  async getSyncStatus() {
    try {
      const response = await fetch(`${this.baseUrl}/api/asl-lex/sync-status`);
      if (!response.ok) {
        throw new Error(`Failed to get sync status: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error getting sync status:', error);
      throw error;
    }
  }

  // Validate Google Sheets URL format
  validateSheetUrl(url) {
    const sheetId = this.extractSheetId(url);
    return sheetId !== null;
  }

  // Get sheet metadata (title, last modified, etc.)
  async getSheetMetadata(sheetId) {
    if (!this.apiKey) {
      throw new Error('Google Sheets API key not configured');
    }

    const url = `${GOOGLE_SHEETS_API_BASE}/${sheetId}?key=${this.apiKey}`;
    
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch sheet metadata: ${response.statusText}`);
      }
      
      const data = await response.json();
      return {
        title: data.properties?.title || 'Untitled',
        lastModified: data.properties?.modifiedTime,
        sheetCount: data.sheets?.length || 0
      };
    } catch (error) {
      console.error('Error fetching sheet metadata:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const googleSheetsClient = new GoogleSheetsClient();
export default googleSheetsClient; 