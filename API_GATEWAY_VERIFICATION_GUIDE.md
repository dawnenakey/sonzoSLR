# API Gateway Verification & Troubleshooting Guide

## 🎯 **Tomorrow's Goals:**
1. **Verify API Gateway endpoints** exist and are properly configured
2. **Fix the 500 Internal Server Error** on index.jsx
3. **Resolve CORS issues** for videos API
4. **Test navigation** on Amplify deployment

---

## 🔍 **API Gateway Endpoints to Verify**

### **Required Endpoints:**
```
GET  /prod/videos           - Fetch video list
GET  /prod/annotations      - Fetch video annotations  
POST /prod/videos           - Upload new video
POST /prod/annotations      - Create new annotation
PUT  /prod/videos/{id}      - Update video metadata
PUT  /prod/annotations/{id} - Update annotation
DELETE /prod/videos/{id}    - Delete video
DELETE /prod/annotations/{id} - Delete annotation
```

### **Authentication Endpoints:**
```
POST /prod/auth/login       - User authentication
POST /prod/auth/register    - User registration
GET  /prod/auth/verify      - Token verification
```

### **Epic 2 & 3 Endpoints:**
```
GET  /prod/text-corpora     - Fetch text corpora
POST /prod/text-corpora     - Create text corpus
GET  /prod/text-segments    - Fetch text segments
POST /prod/video-text-links - Create video-text links
```

---

## 🚨 **Current Issues to Fix**

### **1. 500 Internal Server Error on index.jsx**
- **Error**: Server can't process the file
- **Cause**: Likely import/syntax issues in new page components
- **Location**: `frontend/src/pages/index.jsx`
- **Status**: ❌ **CRITICAL - Blocking deployment**

### **2. CORS Policy Errors**
- **Error**: `Access to fetch at 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/videos' has been blocked by CORS policy`
- **Cause**: API Gateway not allowing requests from `localhost:5173`
- **Status**: ⚠️ **BLOCKING - API calls failing**

### **3. Missing Navigation Items**
- **Issue**: Camera Settings, Camera Test, Troubleshoot, Analysis not showing
- **Cause**: Routing configuration issues
- **Status**: ⚠️ **UI incomplete**

---

## 🔧 **Immediate Fixes Needed**

### **Fix 1: Resolve 500 Error**
```bash
# Check for syntax errors
npm run build

# Check for import issues
grep -r "import.*from" src/pages/

# Verify all page components exist
ls -la src/pages/
```

### **Fix 2: Test Local Build**
```bash
# Start dev server
npm run dev

# Check browser console for errors
# Verify navigation works locally
```

---

## 🌐 **API Gateway Verification Steps**

### **Step 1: Access API Gateway Console**
1. Go to AWS Console → API Gateway
2. Find your API: `qt8f7grhb5.execute-api.us-east-1.amazonaws.com`
3. Check if it's deployed to `prod` stage

### **Step 2: Verify Endpoints**
1. **Resources Tab**: Check if all required endpoints exist
2. **Methods**: Verify GET, POST, PUT, DELETE methods
3. **Integration**: Ensure Lambda functions are properly connected

### **Step 3: Check CORS Configuration**
1. **Gateway Responses**: Look for CORS configuration
2. **Access-Control-Allow-Origin**: Should include your Amplify domain
3. **Access-Control-Allow-Methods**: Should include GET, POST, PUT, DELETE
4. **Access-Control-Allow-Headers**: Should include Content-Type, Authorization

### **Step 4: Test Endpoints**
```bash
# Test videos endpoint
curl -X GET "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/videos"

# Test with CORS headers
curl -H "Origin: https://your-amplify-domain.amplifyapp.com" \
     -H "Access-Control-Request-Method: GET" \
     -X OPTIONS "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod/videos"
```

---

## 📱 **Frontend Testing Checklist**

### **Navigation Verification:**
- [ ] Home page loads with AI Advanced toggle
- [ ] Camera Test page accessible
- [ ] Camera Settings page accessible  
- [ ] Troubleshoot page accessible
- [ ] Analysis page accessible
- [ ] All navigation links work

### **API Functionality:**
- [ ] Videos load without CORS errors
- [ ] Annotations fetch successfully
- [ ] Upload functionality works
- [ ] Search and filtering work

### **Camera Features:**
- [ ] Camera Test page detects devices
- [ ] BRIO camera permissions work
- [ ] OAK camera integration functional
- [ ] Troubleshooting guide accessible

---

## 🐛 **Debugging Commands**

### **Check Build Status:**
```bash
cd frontend
npm run build
```

### **Check for Import Errors:**
```bash
# Find all imports
grep -r "import.*from" src/

# Check specific files
grep -n "import" src/pages/index.jsx
grep -n "import" src/pages/Analysis.jsx
grep -n "import" src/pages/Troubleshoot.jsx
```

### **Check File Structure:**
```bash
# Verify all page files exist
ls -la src/pages/

# Check for syntax errors
node -c src/pages/index.jsx
node -c src/pages/Analysis.jsx
node -c src/pages/Troubleshoot.jsx
```

---

## 📋 **Tomorrow's Action Plan**

### **Morning Session (30 min):**
1. **Fix 500 Internal Server Error** on index.jsx
2. **Test local build** and navigation
3. **Verify all page components** load correctly

### **Afternoon Session (45 min):**
1. **Access AWS API Gateway Console**
2. **Verify all required endpoints** exist
3. **Check CORS configuration** for Amplify domain
4. **Test API endpoints** directly
5. **Deploy to Amplify** and test

### **Evening Session (30 min):**
1. **Test full functionality** on Amplify
2. **Verify camera features** work
3. **Check API calls** succeed
4. **Document any remaining issues**

---

## 🔗 **Useful Links**

- **AWS API Gateway Console**: https://console.aws.amazon.com/apigateway/
- **Amplify Console**: https://console.aws.amazon.com/amplify/
- **Lambda Functions**: https://console.aws.amazon.com/lambda/
- **CloudWatch Logs**: https://console.aws.amazon.com/cloudwatch/

---

## 📞 **When You're Ready Tomorrow:**

1. **Start with the 500 error fix** - this is blocking everything
2. **Test locally first** - make sure navigation works
3. **Then move to API Gateway** - verify endpoints and CORS
4. **Finally deploy to Amplify** - test the full system

**Let me know when you're ready to start!** 🚀
