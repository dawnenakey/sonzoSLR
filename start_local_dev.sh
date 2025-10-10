#!/bin/bash

# SpokHand SLR - Local Development Startup Script
# This script starts all backend services and the frontend for local testing

echo "🚀 Starting SpokHand SLR Local Development Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Port $1 is already in use${NC}"
        echo "Please stop the service using port $1 and try again"
        exit 1
    else
        echo -e "${GREEN}✅ Port $1 is available${NC}"
    fi
}

# Function to start a service
start_service() {
    local service_name=$1
    local port=$2
    local script_path=$3
    
    echo -e "${BLUE}Starting $service_name on port $port...${NC}"
    
    # Check if port is available
    check_port $port
    
    # Start the service in background
    cd src
    python $script_path &
    local pid=$!
    echo $pid > "../pids/${service_name}.pid"
    cd ..
    
    # Wait a moment for service to start
    sleep 2
    
    # Check if service is running
    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✅ $service_name started successfully (PID: $pid)${NC}"
    else
        echo -e "${RED}❌ Failed to start $service_name${NC}"
        exit 1
    fi
}

# Create pids directory
mkdir -p pids

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python is not installed or not in PATH${NC}"
    echo "Please install Python 3.9+ and try again"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed or not in PATH${NC}"
    echo "Please install Node.js 18+ and try again"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed or not in PATH${NC}"
    echo "Please install npm and try again"
    exit 1
fi

echo -e "${YELLOW}📋 Prerequisites check passed${NC}"

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
fi

# Install frontend dependencies if package.json exists
if [ -f "frontend/package.json" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

echo -e "${YELLOW}📦 Dependencies installed${NC}"

# Start backend services
echo -e "${BLUE}🔧 Starting Backend Services...${NC}"

start_service "Authentication Service" 5001 "simple_auth_service.py"
start_service "Text Corpus Service" 5002 "simple_text_corpus_service.py"
start_service "AI Service" 5003 "ai_service.py"
start_service "Lexicon Service" 5004 "lexicon_service.py"
start_service "Analytics Service" 5005 "analytics_service.py"

echo -e "${GREEN}✅ All backend services started${NC}"

# Wait for services to fully initialize
echo -e "${YELLOW}⏳ Waiting for services to initialize...${NC}"
sleep 5

# Test service health
echo -e "${BLUE}🔍 Testing service health...${NC}"

services=(
    "http://localhost:5001/api/auth/health"
    "http://localhost:5002/api/text-corpus/health"
    "http://localhost:5003/api/ai/health"
    "http://localhost:5004/api/lexicon/health"
    "http://localhost:5005/api/analytics/health"
)

for service in "${services[@]}"; do
    if curl -s $service > /dev/null; then
        echo -e "${GREEN}✅ Service is healthy: $service${NC}"
    else
        echo -e "${RED}❌ Service is not responding: $service${NC}"
    fi
done

# Start frontend
echo -e "${BLUE}🌐 Starting Frontend...${NC}"
cd frontend
npm run dev &
frontend_pid=$!
echo $frontend_pid > "../pids/frontend.pid"
cd ..

echo -e "${GREEN}✅ Frontend started (PID: $frontend_pid)${NC}"

# Wait for frontend to start
echo -e "${YELLOW}⏳ Waiting for frontend to initialize...${NC}"
sleep 10

# Check if frontend is running
if ps -p $frontend_pid > /dev/null; then
    echo -e "${GREEN}✅ Frontend is running${NC}"
else
    echo -e "${RED}❌ Frontend failed to start${NC}"
fi

echo ""
echo -e "${GREEN}🎉 SpokHand SLR Development Environment Started Successfully!${NC}"
echo "=================================================="
echo ""
echo -e "${BLUE}📱 Frontend Application:${NC}"
echo "   🌐 Main App: http://localhost:5173"
echo "   📊 Analytics Dashboard: http://localhost:5173/AnalyticsDashboard"
echo "   💼 Investor Presentation: http://localhost:5173/InvestorPresentation"
echo ""
echo -e "${BLUE}🔧 Backend Services:${NC}"
echo "   🔐 Authentication: http://localhost:5001"
echo "   📝 Text Corpus: http://localhost:5002"
echo "   🤖 AI Service: http://localhost:5003"
echo "   📚 Lexicon: http://localhost:5004"
echo "   📊 Analytics: http://localhost:5005"
echo ""
echo -e "${BLUE}🧪 Testing:${NC}"
echo "   📋 Health Checks: ./test_services.sh"
echo "   🧪 Run Tests: ./run_tests.sh"
echo "   📖 Full Guide: ./LOCAL_TESTING_GUIDE.md"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "   • Check the console for any error messages"
echo "   • Use Ctrl+C to stop individual services"
echo "   • Run './stop_all.sh' to stop all services"
echo "   • Check './pids/' directory for process IDs"
echo ""
echo -e "${GREEN}🚀 Ready for testing! Open http://localhost:5173 in your browser${NC}"
