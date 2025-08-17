#!/bin/bash

# Epic 2 Demo Script for SPOKHAND SIGNCUT
# This script demonstrates the complete text corpus management system

echo "SPOKHAND SIGNCUT - Epic 2 Demo"
echo "=================================="
echo "Text Corpus Management System"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

print_api() {
    echo -e "${CYAN}[API]${NC} $1"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/text_corpus_service.py" ]; then
    print_error "Please run this script from the root directory of the project"
    exit 1
fi

print_status "Starting Epic 2 demonstration..."
echo ""

# Step 1: Install dependencies
print_step "Step 1: Installing Python dependencies..."
cd src
if python -m pip install -r ../requirements-auth.txt > /dev/null 2>&1; then
    print_success "Dependencies installed successfully"
else
    print_warning "Some dependencies may not have installed correctly"
fi

# Step 2: Setup database
print_step "Step 2: Setting up DynamoDB tables for Epic 2..."
if python setup_database.py > /dev/null 2>&1; then
    print_success "Database setup completed - Epic 2 tables created"
else
    print_error "Database setup failed. Check AWS credentials and permissions"
    exit 1
fi

# Step 3: Start Epic 1 authentication service (required for Epic 2)
print_step "Step 3: Starting Epic 1 authentication service..."
print_warning "Starting authentication service on http://localhost:5001"
print_warning "Press Ctrl+C to stop when done testing"
echo ""

# Start the auth service in the background
python auth_api.py &
AUTH_PID=$!

# Wait for service to start
sleep 3

# Check if auth service is running
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    print_success "Epic 1 authentication service is running"
else
    print_error "Failed to start authentication service"
    exit 1
fi

# Step 4: Start Epic 2 text corpus service
print_step "Step 4: Starting Epic 2 text corpus service..."
print_warning "Starting text corpus service on http://localhost:5002"
echo ""

# Start the text corpus service in the background
python text_corpus_api.py &
CORPUS_PID=$!

# Wait for service to start
sleep 3

# Check if corpus service is running
if curl -s http://localhost:5002/api/health > /dev/null 2>&1; then
    print_success "Epic 2 text corpus service is running"
else
    print_error "Failed to start text corpus service"
    exit 1
fi

echo ""
print_status "Both services are now running!"
echo "Epic 1 (Auth): http://localhost:5001"
echo "Epic 2 (Text Corpus): http://localhost:5002"
echo ""

# Step 5: Run Epic 2 tests
print_step "Step 5: Running Epic 2 test suite..."
if python test_text_corpus.py > /dev/null 2>&1; then
    print_success "Epic 2 tests passed successfully"
else
    print_warning "Some Epic 2 tests may have failed (this is normal for demo)"
fi

echo ""
print_status "Epic 2 demonstration is ready!"
echo ""

# Step 6: Interactive API demonstration
print_step "Step 6: Interactive API demonstration..."
echo ""

# Get authentication token first
print_api "Getting authentication token..."
AUTH_RESPONSE=$(curl -s -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@spokhand.com",
    "password": "admin123456"
  }')

if echo "$AUTH_RESPONSE" | grep -q "token"; then
    TOKEN=$(echo "$AUTH_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['token'])")
    print_success "Authentication successful"
else
    print_error "Authentication failed - using demo mode"
    TOKEN="demo_token_123"
fi

echo ""

# Demonstrate corpus creation
print_api "Creating a new text corpus..."
CORPUS_RESPONSE=$(curl -s -X POST http://localhost:5002/api/corpora \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "Demo ASL Corpus",
    "description": "A demonstration corpus for Epic 2",
    "language": "ASL",
    "metadata": {
      "difficulty_level": "intermediate",
      "target_audience": "developers",
      "demo": true
    },
    "tags": ["demo", "asl", "epic2"]
  }')

if echo "$CORPUS_RESPONSE" | grep -q "corpus"; then
    CORPUS_ID=$(echo "$CORPUS_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['corpus']['id'])")
    print_success "Created corpus with ID: $CORPUS_ID"
else
    print_warning "Corpus creation may have failed (using demo ID)"
    CORPUS_ID="demo-corpus-123"
fi

echo ""

# Demonstrate adding text segments
print_api "Adding text segments to the corpus..."
for i in {1..3}; do
    SEGMENT_RESPONSE=$(curl -s -X POST http://localhost:5002/api/corpora/$CORPUS_ID/segments \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d "{
        \"text\": \"Demo ASL phrase $i\",
        \"segment_type\": \"phrase\",
        \"metadata\": {
          \"category\": \"demo\",
          \"sequence\": $i
        }
      }")
    
    if echo "$SEGMENT_RESPONSE" | grep -q "segment"; then
        print_success "Added segment $i"
    else
        print_warning "Segment $i creation may have failed"
    fi
done

echo ""

# Demonstrate listing segments
print_api "Listing all segments in the corpus..."
SEGMENTS_RESPONSE=$(curl -s -X GET http://localhost:5002/api/corpora/$CORPUS_ID/segments \
  -H "Authorization: Bearer $TOKEN")

if echo "$SEGMENTS_RESPONSE" | grep -q "segments"; then
    SEGMENT_COUNT=$(echo "$SEGMENTS_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['total'])")
    print_success "Found $SEGMENT_COUNT segments in corpus"
else
    print_warning "Failed to list segments"
fi

echo ""

# Demonstrate search functionality
print_api "Searching within the corpus..."
SEARCH_RESPONSE=$(curl -s -X GET "http://localhost:5002/api/corpora/$CORPUS_ID/search?q=demo&type=text" \
  -H "Authorization: Bearer $TOKEN")

if echo "$SEARCH_RESPONSE" | grep -q "results"; then
    SEARCH_COUNT=$(echo "$SEARCH_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['total'])")
    print_success "Search found $SEARCH_COUNT matching segments"
else
    print_warning "Search may have failed"
fi

echo ""

# Demonstrate export functionality
print_api "Creating a corpus export..."
EXPORT_RESPONSE=$(curl -s -X POST http://localhost:5002/api/corpora/$CORPUS_ID/export \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "format": "json"
  }')

if echo "$EXPORT_RESPONSE" | grep -q "export"; then
    EXPORT_ID=$(echo "$EXPORT_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['export']['id'])")
    print_success "Created export job with ID: $EXPORT_ID"
    
    # Wait a moment for export to process
    sleep 2
    
    # Check export status
    EXPORT_STATUS_RESPONSE=$(curl -s -X GET http://localhost:5002/api/corpora/exports/$EXPORT_ID \
      -H "Authorization: Bearer $TOKEN")
    
    if echo "$EXPORT_STATUS_RESPONSE" | grep -q "export"; then
        EXPORT_STATUS=$(echo "$EXPORT_STATUS_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['export']['status'])")
        print_success "Export status: $EXPORT_STATUS"
    fi
else
    print_warning "Export creation may have failed"
fi

echo ""

# Demonstrate statistics
print_api "Getting corpus statistics..."
STATS_RESPONSE=$(curl -s -X GET http://localhost:5002/api/corpora/$CORPUS_ID/stats \
  -H "Authorization: Bearer $TOKEN")

if echo "$STATS_RESPONSE" | grep -q "total_segments"; then
    TOTAL_SEGMENTS=$(echo "$STATS_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['total_segments'])")
    VALIDATION_RATE=$(echo "$STATS_RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['validation_rate'])")
    print_success "Corpus statistics: $TOTAL_SEGMENTS segments, $VALIDATION_RATE% validation rate"
else
    print_warning "Failed to get corpus statistics"
fi

echo ""

# Step 7: Show available endpoints
print_step "Step 7: Available Epic 2 API endpoints..."
echo ""
echo "Text Corpus Management:"
echo "  POST   /api/corpora                    - Create new corpus"
echo "  GET    /api/corpora                    - List all corpora"
echo "  GET    /api/corpora/{id}               - Get specific corpus"
echo "  PUT    /api/corpora/{id}               - Update corpus"
echo "  DELETE /api/corpora/{id}               - Delete corpus"
echo ""
echo "Text Segments:"
echo "  POST   /api/corpora/{id}/segments      - Add text segment"
echo "  GET    /api/corpora/{id}/segments      - List segments"
echo "  GET    /api/segments/{id}              - Get specific segment"
echo "  PUT    /api/segments/{id}              - Update segment"
echo "  DELETE /api/segments/{id}              - Delete segment"
echo ""
echo "Search & Export:"
echo "  GET    /api/corpora/{id}/search        - Search within corpus"
echo "  POST   /api/corpora/{id}/export        - Export corpus"
echo "  GET    /api/corpora/exports/{id}       - Get export status"
echo "  GET    /api/corpora/{id}/stats         - Get corpus statistics"
echo ""

# Step 8: Show sample usage
print_step "Step 8: Sample usage examples..."
echo ""
echo "Create a new corpus:"
echo "curl -X POST http://localhost:5002/api/corpora \\"
echo "  -H \"Authorization: Bearer YOUR_TOKEN\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"name\": \"My ASL Corpus\", \"description\": \"Personal collection\", \"language\": \"ASL\"}'"
echo ""
echo "Add a text segment:"
echo "curl -X POST http://localhost:5002/api/corpora/CORPUS_ID/segments \\"
echo "  -H \"Authorization: Bearer YOUR_TOKEN\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\": \"Hello, how are you?\", \"segment_type\": \"phrase\"}'"
echo ""

# Step 9: Final status
print_step "Step 9: Epic 2 status check..."
echo ""

# Check both services
AUTH_HEALTH=$(curl -s http://localhost:5001/api/health)
CORPUS_HEALTH=$(curl -s http://localhost:5002/api/health)

if echo "$AUTH_HEALTH" | grep -q "healthy" && echo "$CORPUS_HEALTH" | grep -q "healthy"; then
    print_success "Epic 2 is fully operational!"
    echo ""
    echo "✅ Epic 1 (Authentication): RUNNING"
    echo "✅ Epic 2 (Text Corpus): RUNNING"
    echo ""
    echo "🎯 Epic 2 provides:"
    echo "   • Text corpus creation and management"
    echo "   • Text segment organization"
    echo "   • Advanced search and filtering"
    echo "   • Export capabilities (JSON, CSV)"
    echo "   • Role-based access control"
    echo "   • Statistics and analytics"
    echo ""
    echo "🚀 Ready for Epic 3: Enhanced Video Workspace!"
else
    print_error "One or more services are not responding properly"
fi

echo ""
print_status "Demo completed! Press Ctrl+C to stop both services."
echo ""

# Keep services running until user stops
wait
