# Camera Troubleshooting Guide

## 🎯 **Overview**
This guide helps resolve camera issues for both Logitech BRIO and OAK-D cameras, with special attention to Mac compatibility.

## 🔍 **BRIO Camera Issues on Mac**

### **Common Problem: Camera Access Denied**
**Symptoms:**
- Camera shows black screen
- Error: "Camera access denied" or "Failed to initialize camera"
- Camera works in other apps but not in our application

**Solution:**
1. **Enable Camera Permissions:**
   ```
   System Preferences > Security & Privacy > Privacy > Camera
   ```
   - Check your terminal/IDE (Terminal, VS Code, PyCharm)
   - Check Python if it appears in the list

2. **Restart Terminal/IDE:**
   - Close your terminal or IDE completely
   - Reopen and try again

3. **Check System Integrity Protection:**
   - Some Mac security features may block camera access
   - Try running from Terminal instead of IDE

### **Test BRIO Camera:**
```bash
cd src/camera
python brio_asl.py
```

**Expected Output:**
```
🚀 Testing BRIO ASL Camera...
💻 System: Darwin 23.3.0
🔍 Checking Mac camera permissions...
✅ Mac camera permissions OK
🔍 Searching for BRIO camera...
📹 Camera 0: 1920x1080 @ 30fps
✅ Found BRIO camera at index 0
⚙️  Configuring camera settings...
✅ Camera initialized: 1920x1080 @ 30fps
🎬 Started BRIO camera capture
```

### **If BRIO Still Doesn't Work:**
1. **Check USB Connection:**
   - Try different USB ports
   - Use USB 3.0/3.1 ports for best performance

2. **Update Logitech Software:**
   - Download latest Logitech Camera Settings
   - Reset camera to factory defaults

3. **Test in Other Apps:**
   - FaceTime, Photo Booth, Zoom
   - If it works there, it's a permissions issue

## 🤖 **OAK Camera Functionality**

### **Prerequisites:**
```bash
pip install depthai opencv-python numpy
```

### **Test OAK Camera:**
```bash
cd src/camera
python oak_camera_handler.py
```

**Expected Output:**
```
🚀 Testing OAK Camera Handler...
🔍 Initializing OAK camera...
🔌 Connecting to OAK device...
✅ OAK camera initialized successfully!
📱 Device: OAK-D
🔧 MX ID: [device-id]
📊 State: [state]
🎬 Started OAK camera capture
```

### **OAK Camera Features:**
- **Live Annotation:** Real-time frame capture for annotation
- **High Resolution:** 1080p capture with 640x480 preview
- **Recording:** Built-in video recording capability
- **AI Processing:** Optimized for sign language recognition

### **OAK Camera Controls:**
- **'q'**: Quit
- **'r'**: Start/Stop recording
- **'s'**: Save screenshot

## 🔧 **General Camera Troubleshooting**

### **Check Available Cameras:**
```python
import cv2

# List all available cameras
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {i}: {width}x{height} @ {fps}fps")
        cap.release()
```

### **Camera Index Issues:**
- **Problem:** Camera works at index 1 but not 0
- **Solution:** Our code automatically detects the best camera
- **Fallback:** Uses index 0 if BRIO not found

### **Performance Issues:**
- **Low FPS:** Reduce resolution or check USB bandwidth
- **High CPU:** Close other camera-using applications
- **Memory Issues:** Restart application periodically

## 🚀 **Live Camera Annotation**

### **Both Cameras Support:**
- **Real-time Capture:** Continuous frame streaming
- **Frame Access:** `get_frame()` method for live annotation
- **Recording:** Built-in video recording
- **Integration:** Works with main annotation system

### **API Endpoints:**
```
POST /api/camera/start
{
  "camera_type": "Logitech BRIO" | "OAK-D"
}

POST /api/camera/stop

GET /api/camera/frame
```

### **Frontend Integration:**
- **Live Preview:** Real-time camera feed
- **Annotation Overlay:** Draw annotations on live video
- **Recording Controls:** Start/stop recording
- **Settings Panel:** Adjust camera parameters

## 🐛 **Debugging Steps**

### **Step 1: Check System Info**
```bash
# Mac version
sw_vers

# Python version
python --version

# OpenCV version
python -c "import cv2; print(cv2.__version__)"
```

### **Step 2: Test Camera Access**
```bash
# Test BRIO
cd src/camera
python brio_asl.py

# Test OAK
python oak_camera_handler.py
```

### **Step 3: Check Permissions**
- **Mac:** System Preferences > Security & Privacy > Camera
- **Linux:** Check `/dev/video*` devices
- **Windows:** Check device manager

### **Step 4: Verify Dependencies**
```bash
pip list | grep -E "(opencv|depthai|numpy)"
```

## 📱 **Mac-Specific Solutions**

### **Terminal Camera Access:**
1. **Grant Terminal Camera Permission:**
   - System Preferences > Security & Privacy > Privacy > Camera
   - Check "Terminal" or your terminal app

2. **Run from Terminal:**
   ```bash
   cd /Users/keymuth/spokhandSLR
   python src/camera/brio_asl.py
   ```

3. **Check SIP Status:**
   ```bash
   csrutil status
   ```

### **IDE Camera Access:**
- **VS Code:** May need separate camera permission
- **PyCharm:** Check camera permissions in app settings
- **Jupyter:** Run from terminal instead

## 🔄 **Fallback Options**

### **If BRIO Fails:**
1. **Use Built-in Camera:** Falls back to any available camera
2. **Lower Resolution:** Automatically adjusts to supported settings
3. **Different USB Port:** Try USB 2.0 if 3.0 has issues

### **If OAK Fails:**
1. **Check USB Connection:** Ensure proper USB 3.0 connection
2. **Install DepthAI:** `pip install depthai`
3. **Check Device State:** Verify OAK is powered and connected

## 📞 **Support**

### **Still Having Issues?**
1. **Run Diagnostic:**
   ```bash
   cd src/camera
   python brio_asl.py
   python oak_camera_handler.py
   ```

2. **Check Logs:** Look for error messages and camera info

3. **System Info:** Note your OS version and Python version

4. **Camera Model:** Specify BRIO model and OAK variant

### **Common Solutions:**
- **Restart Application:** Close and reopen
- **Restart Terminal/IDE:** Fresh permission check
- **Check USB Ports:** Try different ports
- **Update Drivers:** Latest Logitech software
- **Reinstall Dependencies:** `pip install --upgrade opencv-python depthai`

## ✅ **Success Indicators**

### **BRIO Working:**
- Camera opens with 1080p or 720p resolution
- 30-60 FPS capture
- Live preview shows video feed
- No permission errors

### **OAK Working:**
- Device connects successfully
- Live video feed displays
- Recording starts/stops
- No connection errors

### **Both Cameras:**
- Seamless switching between camera types
- Live annotation works
- Recording functionality available
- Integration with main system
