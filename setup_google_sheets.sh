#!/bin/bash

# Google Sheets API Setup Script for ASL-LEX Data Management
# This script helps you set up the Google Sheets API integration

echo "🔧 Setting up Google Sheets API for ASL-LEX Data Management"
echo "=========================================================="
echo ""

# Check if API key is already set
if [ -n "$GOOGLE_SHEETS_API_KEY" ]; then
    echo "✅ Google Sheets API key is already configured"
    echo "Current API key: ${GOOGLE_SHEETS_API_KEY:0:10}..."
else
    echo "❌ Google Sheets API key not found"
    echo ""
    echo "📋 To get your Google Sheets API key:"
    echo "1. Go to https://console.cloud.google.com/"
    echo "2. Create a new project or select existing project"
    echo "3. Enable Google Sheets API:"
    echo "   - Go to 'APIs & Services' > 'Library'"
    echo "   - Search for 'Google Sheets API'"
    echo "   - Click 'Enable'"
    echo "4. Create API credentials:"
    echo "   - Go to 'APIs & Services' > 'Credentials'"
    echo "   - Click 'Create Credentials' > 'API Key'"
    echo "   - Copy the API key"
    echo ""
    echo "🔑 Enter your Google Sheets API key:"
    read -s api_key
    echo ""
    
    if [ -n "$api_key" ]; then
        echo "export GOOGLE_SHEETS_API_KEY=\"$api_key\"" >> ~/.bashrc
        echo "export GOOGLE_SHEETS_API_KEY=\"$api_key\"" >> ~/.zshrc
        export GOOGLE_SHEETS_API_KEY="$api_key"
        echo "✅ Google Sheets API key configured"
    else
        echo "❌ No API key provided"
        exit 1
    fi
fi

echo ""
echo "📊 Your Google Sheet Information:"
echo "Sheet ID: 119H0teI02WYil3O6Cl8xt7EiMkm2djSFh1PhrJkoVcg"
echo "Sheet URL: https://docs.google.com/spreadsheets/d/119H0teI02WYil3O6Cl8xt7EiMkm2djSFh1PhrJkoVcg/edit"
echo ""

echo "📋 Required Column Headers for your Google Sheet:"
echo "Gloss | English | Handshape | Location | Movement | Palm Orientation | Dominant Hand | Non-Dominant Hand | Frequency | Age of Acquisition | Iconicity | Lexical Class | Tags | Notes | Video URL | Status | Validation Status | Confidence Score"
echo ""

echo "🚀 Next Steps:"
echo "1. Make sure your Google Sheet has the required headers"
echo "2. Add some test data to your sheet"
echo "3. Start your application"
echo "4. Navigate to the ASL-LEX page"
echo "5. Enter your Google Sheets URL and click 'Sync from Google Sheets'"
echo ""

echo "✅ Setup complete! You can now sync data from your Google Sheet to the ASL-LEX database." 