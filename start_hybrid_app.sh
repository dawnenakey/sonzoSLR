#!/bin/bash

echo "🚀 Starting SpokHand SLR Hybrid Application..."

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $REACT_PID $CAMERA_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Camera Service (Python Backend)
echo "📹 Starting Camera Service..."
cd microservices/camera-service
source venv/bin/activate
python main.py &
CAMERA_PID=$!
cd ../..

# Wait a moment for camera service to start
sleep 3

# Start React Frontend
echo "⚛️  Starting React Frontend..."
cd frontend
npm start &
REACT_PID=$!
cd ..

echo "✅ Services started!"
echo "📹 Camera Service: http://localhost:8001"
echo "⚛️  React Frontend: http://localhost:3000"
echo "📖 API Documentation: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait 