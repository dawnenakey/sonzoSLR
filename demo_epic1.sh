#!/bin/bash

# Epic 1 Demo Script for SPOKHAND SIGNCUT
# This script demonstrates the complete authentication system

echo "SPOKHAND SIGNCUT - Epic 1 Demo"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/auth_service.py" ]; then
    print_error "Please run this script from the root directory of the project"
    exit 1
fi

print_status "Starting Epic 1 demonstration..."

# Step 1: Install dependencies
print_status "Step 1: Installing Python dependencies..."
cd src
if python -m pip install -r ../requirements-auth.txt > /dev/null 2>&1; then
    print_success "Dependencies installed successfully"
else
    print_warning "Some dependencies may not have installed correctly"
fi

# Step 2: Setup database
print_status "Step 2: Setting up DynamoDB tables..."
if python setup_database.py > /dev/null 2>&1; then
    print_success "Database setup completed"
else
    print_error "Database setup failed. Check AWS credentials and permissions"
    exit 1
fi

# Step 3: Start the service
print_status "Step 3: Starting authentication service..."
print_warning "Starting service on http://localhost:5001"
print_warning "Press Ctrl+C to stop the service when done testing"
echo ""

# Start the service in the background
python auth_api.py &
SERVICE_PID=$!

# Wait for service to start
sleep 3

# Check if service is running
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    print_success "Service is running and responding"
else
    print_error "Service failed to start properly"
    kill $SERVICE_PID 2>/dev/null
    exit 1
fi

echo ""
print_status "Demo Scenarios Available:"
echo ""

# Demo 1: User Registration
print_status "Demo 1: User Registration"
echo "curl -X POST http://localhost:5001/api/auth/register \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"email\": \"demo@spokhand.com\", \"password\": \"demo123456\", \"full_name\": \"Demo User\"}'"
echo ""

# Demo 2: User Login
print_status "Demo 2: User Login"
echo "curl -X POST http://localhost:5001/api/auth/login \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"email\": \"demo@spokhand.com\", \"password\": \"demo123456\"}'"
echo ""

# Demo 3: Protected Endpoint
print_status "Demo 3: Protected Endpoint (replace YOUR_TOKEN)"
echo "curl -X GET http://localhost:5001/api/auth/me \\"
echo "  -H \"Authorization: Bearer YOUR_JWT_TOKEN_HERE\""
echo ""

# Demo 4: Admin Access
print_status "Demo 4: Admin Access (use admin@spokhand.com / admin123456)"
echo "curl -X GET http://localhost:5001/api/admin/users \\"
echo "  -H \"Authorization: Bearer ADMIN_JWT_TOKEN\""
echo ""

# Demo 5: Role-Based Access
print_status "Demo 5: Role-Based Access Control"
echo "Login as translator@spokhand.com / translator123456"
echo "Try to access admin endpoint - should get 403 Forbidden"
echo ""

# Demo 6: Security Features
print_status "Demo 6: Security Features"
echo "Invalid token: curl -X GET http://localhost:5001/api/auth/me -H \"Authorization: Bearer invalid_token\""
echo "Missing token: curl -X GET http://localhost:5001/api/auth/me"
echo ""

# Demo 7: Frontend
print_status "Demo 7: Frontend Integration"
echo "1. Update your App.tsx with AuthProvider"
echo "2. Add /auth route to your router"
echo "3. Navigate to /auth to see login/register forms"
echo ""

# Demo 8: Run Tests
print_status "Demo 8: Automated Testing"
echo "In another terminal, run:"
echo "cd src && python test_auth.py"
echo ""

print_success "Epic 1 is ready for demonstration!"
echo ""
print_status "Sample users created:"
echo "  • admin@spokhand.com / admin123456 (Admin)"
echo "  • expert@spokhand.com / expert123456 (Expert+)"
echo "  • qualifier@spokhand.com / qualifier123456 (Qualifier+)"
echo "  • segmenter@spokhand.com / segmenter123456 (Segmenter+)"
echo "  • translator@spokhand.com / translator123456 (Translator)"
echo ""

print_status "Service is running on http://localhost:5001"
print_status "Press Ctrl+C to stop the service"
echo ""

# Wait for user to stop the service
trap "echo ''; print_status 'Stopping service...'; kill $SERVICE_PID 2>/dev/null; print_success 'Demo completed!'; exit 0" INT

# Keep the service running
wait $SERVICE_PID 