# SpokHand SLR - Hybrid Application Setup

## 🎯 **What We've Built**

A hybrid microservices application with:
- **React Frontend** (TypeScript) - Unified UI
- **Python Backend** (FastAPI) - Camera and ML services
- **Real-time Communication** - WebSocket for camera feed
- **Sign Vocabulary** - Organized sign categories

## 📁 **Project Structure**

```
spokhandSLR/
├── frontend/                    # React TypeScript app
│   ├── src/
│   │   ├── components/
│   │   │   └── CameraInterface.tsx  # Main camera UI
│   │   └── App.tsx
│   └── package.json
├── microservices/
│   └── camera-service/          # Python FastAPI service
│       ├── main.py             # Camera API endpoints
│       ├── requirements.txt    # Python dependencies
│       └── venv/               # Virtual environment
├── start_hybrid_app.sh         # Startup script
└── docs/                       # Architecture documentation
```

## 🚀 **Quick Start**

### **Option 1: Automated Startup (Recommended)**
```bash
./start_hybrid_app.sh
```

### **Option 2: Manual Startup**

**Terminal 1 - Start Camera Service:**
```bash
cd microservices/camera-service
source venv/bin/activate
python main.py
```

**Terminal 2 - Start React Frontend:**
```bash
cd frontend
npm start
```

## 🌐 **Access Points**

- **React Frontend**: http://localhost:3000
- **Camera Service API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## 🎮 **How to Use**

1. **Open** http://localhost:3000 in your browser
2. **Select** camera type (BRIO or OAK-D)
3. **Click** "Start Camera" to initialize
4. **Choose** a sign category and specific sign
5. **Click** "Start Recording" to record the sign
6. **Click** "Stop Recording" when done

## 🔧 **Features Implemented**

### ✅ **Frontend (React)**
- Camera type selection
- Sign vocabulary with categories
- Real-time status display
- Recording controls
- Session management

### ✅ **Backend (Python)**
- FastAPI REST endpoints
- Camera session management
- Video recording
- MediaPipe hand tracking
- WebSocket support

### ✅ **Integration**
- CORS configuration
- API communication
- Error handling
- Session state management

## 📊 **API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/camera/start` | Start camera session |
| POST | `/api/camera/stop` | Stop camera session |
| POST | `/api/camera/record` | Start recording |
| POST | `/api/camera/stop-recording` | Stop recording |
| GET | `/api/camera/sessions` | List active sessions |
| WS | `/ws/camera/{session_id}` | Real-time camera feed |

## 🎯 **Next Steps**

### **Phase 1: Complete MVP (This Week)**
- [ ] Add real-time video feed display
- [ ] Integrate annotation service
- [ ] Add file upload functionality
- [ ] Implement basic error handling

### **Phase 2: Advanced Features (Next Week)**
- [ ] Add storage service
- [ ] Implement analytics
- [ ] Add user authentication
- [ ] Create deployment pipeline

### **Phase 3: Production Ready (Week 3)**
- [ ] Add monitoring and logging
- [ ] Implement security features
- [ ] Performance optimization
- [ ] Documentation completion

## 🛠️ **Development Commands**

### **Frontend Development**
```bash
cd frontend
npm start          # Start development server
npm run build      # Build for production
npm test           # Run tests
```

### **Backend Development**
```bash
cd microservices/camera-service
source venv/bin/activate
python main.py     # Start camera service
```

### **API Testing**
```bash
# Test camera service
curl -X POST http://localhost:8001/api/camera/start \
  -H "Content-Type: application/json" \
  -d '{"camera_type": "brio"}'
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Camera not starting**
   - Check if camera is connected
   - Verify camera permissions
   - Check port 8001 is available

2. **React not loading**
   - Check if port 3000 is available
   - Verify Node.js version (18+)
   - Check npm dependencies

3. **CORS errors**
   - Verify CORS configuration in camera service
   - Check frontend URL matches allowed origins

### **Logs**
- **Camera Service**: Check terminal running `python main.py`
- **React Frontend**: Check browser console and terminal

## 📈 **Current Status**

- ✅ **Project Structure**: Complete
- ✅ **Basic Integration**: Working
- ✅ **Camera Service**: Functional
- ✅ **React Frontend**: Functional
- 🟡 **Real-time Video**: In Progress
- 🔲 **Annotation Integration**: Pending
- 🔲 **Storage Service**: Pending

## 🎉 **Success!**

You now have a working hybrid application with:
- React frontend for UI
- Python backend for camera/ML
- API communication between them
- Sign vocabulary system
- Recording capabilities

**Ready to continue with the next phase!** 