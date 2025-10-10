#!/bin/bash

# SpokHand SLR - Service Health Check Script
# This script tests all services to ensure they're running correctly

echo "🧪 Testing SpokHand SLR Services"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test a service
test_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -e "${BLUE}Testing $service_name...${NC}"
    
    # Make request with timeout
    response=$(curl -s -w "%{http_code}" -o /dev/null --max-time 10 "$url" 2>/dev/null)
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✅ $service_name is healthy (HTTP $response)${NC}"
        return 0
    else
        echo -e "${RED}❌ $service_name is not responding (HTTP $response)${NC}"
        return 1
    fi
}

# Test all services
echo -e "${BLUE}🔍 Running health checks...${NC}"
echo ""

# Backend services
test_service "Authentication Service" "http://localhost:5001/api/auth/health"
test_service "Text Corpus Service" "http://localhost:5002/api/text-corpus/health"
test_service "AI Service" "http://localhost:5003/api/ai/health"
test_service "Lexicon Service" "http://localhost:5004/api/lexicon/health"
test_service "Analytics Service" "http://localhost:5005/api/analytics/health"

echo ""

# Test frontend
echo -e "${BLUE}Testing Frontend...${NC}"
if curl -s --max-time 10 "http://localhost:5173" > /dev/null; then
    echo -e "${GREEN}✅ Frontend is accessible${NC}"
else
    echo -e "${RED}❌ Frontend is not accessible${NC}"
fi

echo ""

# Test specific API endpoints
echo -e "${BLUE}🔍 Testing API Endpoints...${NC}"

# Test authentication endpoints
echo -e "${YELLOW}Testing Authentication API...${NC}"
curl -s -X GET "http://localhost:5001/api/auth/health" | jq . 2>/dev/null || echo "Response received (jq not available)"

# Test analytics dashboard data
echo -e "${YELLOW}Testing Analytics API...${NC}"
curl -s -X GET "http://localhost:5005/api/analytics/dashboard" | jq . 2>/dev/null || echo "Response received (jq not available)"

# Test lexicon API
echo -e "${YELLOW}Testing Lexicon API...${NC}"
curl -s -X GET "http://localhost:5004/api/lexicon/signs" | jq . 2>/dev/null || echo "Response received (jq not available)"

echo ""
echo -e "${GREEN}🎉 Service testing complete!${NC}"
echo ""
echo -e "${BLUE}📱 Access URLs:${NC}"
echo "   🌐 Frontend: http://localhost:5173"
echo "   📊 Analytics Dashboard: http://localhost:5173/AnalyticsDashboard"
echo "   💼 Investor Presentation: http://localhost:5173/InvestorPresentation"
echo ""
echo -e "${BLUE}🔧 Backend APIs:${NC}"
echo "   🔐 Authentication: http://localhost:5001"
echo "   📝 Text Corpus: http://localhost:5002"
echo "   🤖 AI Service: http://localhost:5003"
echo "   📚 Lexicon: http://localhost:5004"
echo "   📊 Analytics: http://localhost:5005"
