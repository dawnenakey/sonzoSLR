#!/bin/bash

# SpokHand SLR - Stop All Services Script
# This script stops all running services

echo "🛑 Stopping SpokHand SLR Services"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to stop a service
stop_service() {
    local service_name=$1
    local pid_file="pids/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null; then
            echo -e "${BLUE}Stopping $service_name (PID: $pid)...${NC}"
            kill $pid
            sleep 2
            if ps -p $pid > /dev/null; then
                echo -e "${YELLOW}Force killing $service_name...${NC}"
                kill -9 $pid
            fi
            echo -e "${GREEN}✅ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $service_name was not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  No PID file found for $service_name${NC}"
    fi
}

# Stop all services
echo -e "${BLUE}🔧 Stopping Backend Services...${NC}"
stop_service "Authentication Service"
stop_service "Text Corpus Service"
stop_service "AI Service"
stop_service "Lexicon Service"
stop_service "Analytics Service"

echo -e "${BLUE}🌐 Stopping Frontend...${NC}"
stop_service "frontend"

# Kill any remaining Python processes related to our services
echo -e "${BLUE}🧹 Cleaning up remaining processes...${NC}"
pkill -f "auth_service.py" 2>/dev/null || true
pkill -f "text_corpus_service.py" 2>/dev/null || true
pkill -f "ai_service.py" 2>/dev/null || true
pkill -f "lexicon_service.py" 2>/dev/null || true
pkill -f "analytics_service.py" 2>/dev/null || true

# Clean up pids directory
rm -rf pids/

echo -e "${GREEN}✅ All services stopped successfully${NC}"
echo ""
echo -e "${YELLOW}💡 Note: If you see any 'process not found' messages, that's normal - it means the service was already stopped${NC}"
