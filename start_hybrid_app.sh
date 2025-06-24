#!/bin/bash

echo "🚀 Starting SpokHand SLR Hybrid Application..."

# Function to kill processes on specified ports
cleanup() {
    echo "Shutting down services..."
    kill $(lsof -t -i:5173) 2>/dev/null
    echo "Cleanup complete."
}

# Trap exit signals to run cleanup
trap cleanup EXIT

# Source NVM to make it available to the script
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Start React frontend
echo "Starting React frontend..."
(cd frontend && nvm use stable && yarn start --host) &
FRONTEND_PID=$!

# Wait for all background processes to complete
wait $FRONTEND_PID

echo "✅ Services started!"
echo "⚛️  React Frontend: http://localhost:5173"
echo "📖 API Documentation: http://localhost:8000/docs and http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes to complete
wait $FRONTEND_PID 