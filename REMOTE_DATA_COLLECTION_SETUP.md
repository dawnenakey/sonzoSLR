# 🎥 SpokHand SLR Remote Data Collection Setup

This guide will help you set up a **remote data collection system** for sign language research using AWS infrastructure and the Logitech BRIO camera.

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Analyst  │    │   React Frontend │    │   AWS Backend   │
│   (Remote)      │◄──►│   (Local/Cloud)  │◄──►│   (Cloud)       │
│                 │    │                 │    │                 │
│ • BRIO Camera   │    │ • Video Upload  │    │ • S3 Storage    │
│ • Web Browser   │    │ • Session Mgmt  │    │ • DynamoDB      │
│ • Recording UI  │    │ • Real-time     │    │ • Lambda        │
└─────────────────┘    │   Processing    │    │ • CloudFront    │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start**

### 1. **Deploy AWS Infrastructure**

```bash
# Deploy the remote data collection infrastructure
./scripts/deploy-data-collection.sh dev us-east-1
```

This will create:
- **S3 Bucket**: For video storage
- **DynamoDB Table**: For session and video metadata
- **API Gateway**: For remote access
- **Lambda Functions**: For video processing
- **CloudFront**: For video streaming

### 2. **Start the Data Collection Service**

```bash
cd microservices/data-collection-service

# Install dependencies
pip install -r requirements.txt

# Start the service
python main.py
```

The service will run on `http://localhost:8002`

### 3. **Update Frontend Environment**

Add to `frontend/.env`:
```env
REACT_APP_DATA_COLLECTION_API=http://localhost:8002
```

### 4. **Start the Frontend**

```bash
cd frontend
npm start
```

## 📋 **Features**

### **For Data Analysts (Remote Users)**

✅ **Session Management**
- Create named data collection sessions
- Add descriptions and tags
- Track video count and duration

✅ **Video Recording**
- High-quality video capture (up to 4K)
- Real-time camera preview
- Recording timer and controls
- Automatic hand landmark detection

✅ **Upload & Processing**
- Secure video upload to AWS S3
- Background video processing
- Hand landmark detection using MediaPipe
- Progress tracking

✅ **Video Management**
- View all session videos
- Check processing status
- Stream videos remotely
- Metadata tracking

### **For Researchers**

✅ **Data Organization**
- Structured session management
- Tagged video collections
- User-based organization
- Search and filter capabilities

✅ **Quality Control**
- Automatic hand detection validation
- Video metadata extraction
- Processing status monitoring
- Error handling and logging

✅ **Scalability**
- Cloud-based storage
- CDN for video streaming
- Serverless processing
- Multi-user support

## 🔧 **Technical Details**

### **Backend Services**

#### **Data Collection Service** (`microservices/data-collection-service/`)
- **FastAPI** web framework
- **MediaPipe** hand landmark detection
- **OpenCV** video processing
- **AWS SDK** integration
- **Background task processing**

#### **API Endpoints**
```
POST /sessions/create          # Create new session
POST /sessions/{id}/upload     # Upload video
GET  /sessions/{id}/videos     # Get session videos
GET  /sessions/user/{user_id}  # Get user sessions
GET  /videos/{id}/stream       # Stream video
```

### **Frontend Components**

#### **RemoteDataCollection** (`frontend/src/components/RemoteDataCollection.tsx`)
- **React** component for data collection
- **MediaRecorder API** for video capture
- **Real-time** camera preview
- **Session management** interface
- **Upload progress** tracking

### **AWS Infrastructure**

#### **Storage**
- **S3 Bucket**: Video file storage with lifecycle policies
- **DynamoDB**: Session and video metadata
- **CloudFront**: Video streaming CDN

#### **Processing**
- **Lambda**: Serverless video processing
- **MediaPipe**: Hand landmark detection
- **OpenCV**: Video metadata extraction

## 🎯 **Usage Workflow**

### **For Data Analysts**

1. **Access the System**
   - Open the React frontend in a web browser
   - Navigate to the Remote Data Collection section

2. **Create a Session**
   - Click "Create New Session"
   - Enter session name, description, and tags
   - Session is created in AWS DynamoDB

3. **Start Recording**
   - Select the session from the list
   - Click "Start Camera" to access BRIO camera
   - Click "Start Recording" to begin capture
   - Record sign language videos

4. **Upload Videos**
   - Click "Stop Recording" when done
   - Video automatically uploads to AWS S3
   - Background processing extracts metadata
   - Hand landmarks are detected automatically

5. **Monitor Progress**
   - View upload progress in real-time
   - Check processing status for each video
   - Verify hand landmark detection results

### **For Researchers**

1. **Review Sessions**
   - Access the admin interface
   - View all data collection sessions
   - Filter by user, date, or tags

2. **Quality Control**
   - Review video metadata
   - Check hand landmark detection results
   - Validate video quality and content

3. **Data Export**
   - Download videos for analysis
   - Export session metadata
   - Generate collection reports

## 🔒 **Security & Privacy**

### **Data Protection**
- **Encrypted storage** in S3
- **Secure API endpoints** with authentication
- **Private video access** via presigned URLs
- **User-based access control**

### **Privacy Compliance**
- **Local processing** of sensitive video data
- **Secure transmission** to AWS
- **Data retention policies**
- **User consent management**

## 📊 **Monitoring & Analytics**

### **System Health**
- **Service health checks**
- **Error logging and alerting**
- **Performance monitoring**
- **Usage analytics**

### **Data Quality**
- **Video processing success rates**
- **Hand landmark detection accuracy**
- **Upload completion rates**
- **User session statistics**

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Camera Access Denied**
```bash
# Check browser permissions
# Ensure HTTPS for production
# Verify camera is not in use by other applications
```

#### **Upload Failures**
```bash
# Check AWS credentials
# Verify S3 bucket permissions
# Check network connectivity
# Review file size limits
```

#### **Processing Errors**
```bash
# Check Lambda function logs
# Verify MediaPipe installation
# Review video format compatibility
# Check DynamoDB permissions
```

### **Debug Commands**

```bash
# Check service health
curl http://localhost:8002/health

# Test AWS connectivity
aws s3 ls s3://your-bucket-name

# Check DynamoDB table
aws dynamodb scan --table-name your-table-name

# View CloudWatch logs
aws logs describe-log-groups
```

## 🔄 **Integration with Base44**

Once Base44 API credentials are available:

1. **Replace mock client** with real Base44 SDK
2. **Upload videos** directly to Base44 platform
3. **Sync metadata** between systems
4. **Enable advanced annotation** features

## 📈 **Scaling Considerations**

### **Performance**
- **CDN caching** for video streaming
- **Lambda concurrency** for processing
- **Database indexing** for queries
- **Load balancing** for API endpoints

### **Cost Optimization**
- **S3 lifecycle policies** for storage
- **Lambda timeout** optimization
- **CloudFront caching** strategies
- **DynamoDB capacity** planning

## 🎉 **Next Steps**

1. **Deploy the infrastructure** using the provided script
2. **Test the system** with sample videos
3. **Train data analysts** on the workflow
4. **Monitor usage** and optimize performance
5. **Integrate with Base44** when credentials are available

---

**Need Help?** Check the troubleshooting section or contact the development team. 