# 🧪 SpokHand SLR - Local Testing Guide

## **Complete Local Development Setup**

This guide will help you test the entire SpokHand SLR platform locally, including all 6 completed epics and the new analytics dashboard.

---

## 📋 **Prerequisites**

### **Required Software**
- **Node.js** (v18+): [Download here](https://nodejs.org/)
- **Python** (v3.9+): [Download here](https://python.org/)
- **Git**: [Download here](https://git-scm.com/)
- **AWS CLI**: [Install guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

### **AWS Setup**
```bash
# Configure AWS CLI
aws configure

# Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

---

## 🚀 **Quick Start (5 Minutes)**

### **1. Clone and Setup**
```bash
# Clone the repository
git clone <repository-url>
cd spokhandSLR

# Install frontend dependencies
cd frontend
npm install
cd ..

# Install Python dependencies
pip install -r requirements.txt
```

### **2. Start All Services**
```bash
# Terminal 1: Start Authentication Service
cd src
python auth_service.py

# Terminal 2: Start Text Corpus Service
python text_corpus_service.py

# Terminal 3: Start AI Service
python ai_service.py

# Terminal 4: Start Lexicon Service
python lexicon_service.py

# Terminal 5: Start Analytics Service
python analytics_service.py

# Terminal 6: Start Frontend
cd frontend
npm run dev
```

### **3. Access the Platform**
- **Frontend**: http://localhost:5173
- **Analytics Dashboard**: http://localhost:5173/AnalyticsDashboard
- **Investor Presentation**: http://localhost:5173/InvestorPresentation

---

## 🔧 **Detailed Setup Instructions**

### **Backend Services Setup**

#### **1. Authentication Service (Port 5001)**
```bash
cd src
python auth_service.py
```
**Test**: http://localhost:5001/api/auth/health

#### **2. Text Corpus Service (Port 5002)**
```bash
python text_corpus_service.py
```
**Test**: http://localhost:5002/api/text-corpus/health

#### **3. AI Service (Port 5003)**
```bash
python ai_service.py
```
**Test**: http://localhost:5003/api/ai/health

#### **4. Lexicon Service (Port 5004)**
```bash
python lexicon_service.py
```
**Test**: http://localhost:5004/api/lexicon/health

#### **5. Analytics Service (Port 5005)**
```bash
python analytics_service.py
```
**Test**: http://localhost:5005/api/analytics/health

### **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```
**Access**: http://localhost:5173

---

## 🧪 **Testing Each Epic**

### **Epic 1: Authentication Testing**
```bash
# Test user registration
curl -X POST http://localhost:5001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com", "password": "password123"}'

# Test user login
curl -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'
```

### **Epic 2: Text Corpus Testing**
```bash
# Test corpus creation
curl -X POST http://localhost:5002/api/text-corpus/corpus \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Corpus", "description": "Test description", "language": "en"}'

# Test segment creation
curl -X POST http://localhost:5002/api/text-corpus/segments \
  -H "Content-Type: application/json" \
  -d '{"corpus_id": "corpus_id", "text": "Test segment", "start_time": 0, "end_time": 5}'
```

### **Epic 3: Video Workspace Testing**
1. Navigate to http://localhost:5173/Annotator
2. Upload a test video file
3. Create annotations on the timeline
4. Test playback and annotation editing

### **Camera Integration Testing**
1. **Navigate to Camera Settings**: http://localhost:5173/CameraSettings
2. **Test Camera Detection**: Verify OAK-D, BRIO, iPhone, Android, and other cameras are detected
3. **Test Camera Selection**: Select different camera types
4. **Test Camera Preview**: Start camera and verify video stream
5. **Test Camera Settings**: Adjust resolution, frame rate, and quality
6. **Test Camera Test Utility**: Run comprehensive camera tests

### **Mobile Camera Testing**
1. **iOS Testing**:
   - Open Safari on iPhone/iPad
   - Navigate to http://localhost:5173 (or your server IP)
   - Test camera detection and selection
   - Test front/back camera switching
   - Test landscape/portrait orientation
   - Verify video quality and performance

2. **Android Testing**:
   - Open Chrome on Android device
   - Navigate to http://localhost:5173 (or your server IP)
   - Test camera detection and selection
   - Test front/back camera switching
   - Test landscape/portrait orientation
   - Verify video quality and performance

3. **Mobile-Specific Features**:
   - Test touch controls and gestures
   - Test responsive design on different screen sizes
   - Test battery usage during recording
   - Test network performance on mobile data

### **Epic 4: AI Integration Testing**
```bash
# Test AI analysis
curl -X POST http://localhost:5003/api/ai/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"video_id": "test_video", "fusion_strategy": "late_fusion"}'

# Test export results
curl -X GET http://localhost:5003/api/ai/export-results/test_video
```

### **Epic 5: Lexicon Management Testing**
```bash
# Test sign creation
curl -X POST http://localhost:5004/api/lexicon/signs \
  -H "Content-Type: application/json" \
  -d '{"gloss": "HELLO", "english": "Hello", "handshape": "A", "location": "neutral"}'

# Test sign listing
curl -X GET http://localhost:5004/api/lexicon/signs
```

### **Epic 6: Analytics Testing**
```bash
# Test dashboard data
curl -X GET http://localhost:5005/api/analytics/dashboard

# Test system metrics
curl -X GET http://localhost:5005/api/analytics/system-metrics

# Test export report
curl -X GET http://localhost:5005/api/analytics/export-report?type=executive&format=json
```

---

## 🎯 **Frontend Testing**

### **Main Pages**
- **Home**: http://localhost:5173/
- **Annotator**: http://localhost:5173/Annotator
- **Lexicon**: http://localhost:5173/Lexicon
- **Segments**: http://localhost:5173/Segments
- **Analysis**: http://localhost:5173/Analysis
- **ASL-LEX**: http://localhost:5173/ASLLex

### **New Analytics Pages**
- **Analytics Dashboard**: http://localhost:5173/AnalyticsDashboard
- **Investor Presentation**: http://localhost:5173/InvestorPresentation

### **Testing Checklist**
- [ ] All pages load without errors
- [ ] Navigation works between pages
- [ ] Forms submit successfully
- [ ] Data displays correctly
- [ ] Real-time updates work
- [ ] Export functions work
- [ ] Responsive design on mobile
- [ ] Camera detection works (OAK-D, BRIO, iPhone, Android, standard)
- [ ] Camera selection and preview function
- [ ] Camera settings save and load correctly
- [ ] Camera test utility works properly
- [ ] Mobile camera detection works (iOS/Android)
- [ ] Mobile camera switching works (front/back)
- [ ] Mobile responsive design works on different screen sizes
- [ ] Mobile touch controls and gestures work properly

---

## 🔍 **API Testing with Postman**

### **Import Collection**
1. Download the Postman collection from `/docs/postman/`
2. Import into Postman
3. Set environment variables:
   - `base_url`: http://localhost
   - `auth_token`: (get from login)

### **Test All Endpoints**
- Authentication: 8 endpoints
- Text Corpus: 15 endpoints
- AI Service: 6 endpoints
- Lexicon: 8 endpoints
- Analytics: 7 endpoints

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Port Already in Use**
```bash
# Find process using port
lsof -i :5001

# Kill process
kill -9 <PID>
```

#### **AWS Credentials Error**
```bash
# Check AWS configuration
aws sts get-caller-identity

# Reconfigure if needed
aws configure
```

#### **Frontend Build Errors**
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

#### **Python Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### **Service Health Checks**
```bash
# Check all services
curl http://localhost:5001/api/auth/health
curl http://localhost:5002/api/text-corpus/health
curl http://localhost:5003/api/ai/health
curl http://localhost:5004/api/lexicon/health
curl http://localhost:5005/api/analytics/health
```

---

## 📊 **Performance Testing**

### **Load Testing**
```bash
# Install Apache Bench
brew install httpd  # macOS
# or
sudo apt-get install apache2-utils  # Ubuntu

# Test API endpoints
ab -n 1000 -c 10 http://localhost:5001/api/auth/health
ab -n 1000 -c 10 http://localhost:5005/api/analytics/dashboard
```

### **Memory Usage**
```bash
# Monitor Python processes
ps aux | grep python

# Monitor Node.js processes
ps aux | grep node
```

---

## 🎉 **Demo Scenarios**

### **Investor Demo (15 minutes)**
1. **Start**: Show Analytics Dashboard
2. **AI Demo**: Upload video, run analysis
3. **Lexicon**: Show sign management
4. **Analytics**: Show real-time metrics
5. **Presentation**: Run investor slideshow

### **Technical Demo (30 minutes)**
1. **Architecture**: Show all services running
2. **API Testing**: Demonstrate all endpoints
3. **Frontend**: Show all pages and features
4. **Data Flow**: Show data between services
5. **Performance**: Show metrics and monitoring

---

## 📝 **Testing Checklist**

### **Backend Services**
- [ ] All 5 services start without errors
- [ ] Health checks return 200 OK
- [ ] Database connections work
- [ ] API endpoints respond correctly
- [ ] Error handling works
- [ ] Logging is working

### **Frontend Application**
- [ ] All pages load correctly
- [ ] Navigation works smoothly
- [ ] Forms submit successfully
- [ ] Real-time updates work
- [ ] Export functions work
- [ ] Mobile responsive design

### **Integration Testing**
- [ ] Frontend connects to all backend services
- [ ] Data flows correctly between services
- [ ] Authentication works across all services
- [ ] Analytics data updates in real-time
- [ ] AI analysis results display correctly

### **Performance Testing**
- [ ] Services handle concurrent requests
- [ ] Response times are acceptable
- [ ] Memory usage is reasonable
- [ ] No memory leaks detected

---

## 🚀 **Production Deployment**

### **AWS Deployment**
```bash
# Deploy to AWS Lambda
serverless deploy

# Deploy frontend to S3
npm run build
aws s3 sync dist/ s3://your-bucket-name
```

### **Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up --build
```

---

## 📞 **Support**

### **Getting Help**
- **Documentation**: Check `/docs/` folder
- **API Docs**: http://localhost:5001/docs (when running)
- **Logs**: Check console output for errors
- **Issues**: Create GitHub issue with error details

### **Quick Commands**
```bash
# Restart all services
./scripts/restart_all.sh

# Check service status
./scripts/check_services.sh

# Run all tests
./scripts/run_tests.sh
```

---

**🎯 The SpokHand SLR platform is now ready for comprehensive local testing with all 6 epics complete and fully functional!**
