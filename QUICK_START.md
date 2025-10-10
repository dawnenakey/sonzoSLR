# 🚀 SpokHand SLR - Quick Start Guide

## **Get Up and Running in 5 Minutes**

### **Prerequisites**
- Node.js 18+ and Python 3.9+ installed
- AWS CLI configured (optional for full functionality)

### **1. Start Everything**
```bash
# Make scripts executable (if not already done)
chmod +x start_local_dev.sh stop_all.sh test_services.sh

# Start all services
./start_local_dev.sh
```

### **2. Access the Platform**
- **Main App**: http://localhost:5173
- **Analytics Dashboard**: http://localhost:5173/AnalyticsDashboard  
- **Investor Presentation**: http://localhost:5173/InvestorPresentation

### **3. Test Services**
```bash
# Check if all services are running
./test_services.sh
```

### **4. Stop Everything**
```bash
# Stop all services
./stop_all.sh
```

---

## **What You'll See**

### **Analytics Dashboard** (`/AnalyticsDashboard`)
- Real-time system metrics
- Business intelligence data
- AI performance statistics
- User engagement analytics
- Export capabilities

### **Investor Presentation** (`/InvestorPresentation`)
- Interactive slideshow
- Market opportunity analysis
- Technology showcase
- Business metrics
- Investment details

### **Main Platform Features**
- **Authentication**: User management and security
- **Video Annotation**: Advanced timeline-based annotation
- **AI Analysis**: Sign language recognition
- **Lexicon Management**: ASL sign database
- **Text Corpus**: Text data management

---

## **Troubleshooting**

### **Port Already in Use**
```bash
# Find and kill process using port
lsof -i :5001
kill -9 <PID>
```

### **Services Not Starting**
```bash
# Check logs in terminal output
# Install missing dependencies
pip install -r requirements.txt
cd frontend && npm install
```

### **Frontend Not Loading**
```bash
# Restart frontend
cd frontend
npm run dev
```

---

## **Full Documentation**
- **Complete Testing Guide**: `LOCAL_TESTING_GUIDE.md`
- **MVP Summary**: `MVP_COMPLETION_SUMMARY.md`
- **API Documentation**: Available when services are running

---

**🎉 You're ready to demo the complete SpokHand SLR platform to investors!**
