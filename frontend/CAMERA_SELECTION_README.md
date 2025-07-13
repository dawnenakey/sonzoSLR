# Camera Selection Features for SpokHand SLR

This document describes the new camera selection and optimization features added to the SpokHand SLR application for data analysts.

## 🎥 Overview

The application now includes comprehensive camera selection and optimization features, with special support for Logitech BRIO cameras and other high-quality recording devices.

## ✨ New Features

### 1. Camera Selector Component
- **Automatic BRIO Detection**: Automatically detects and optimizes settings for Logitech BRIO cameras
- **Multi-Camera Support**: Supports multiple camera devices with automatic switching
- **Smart Defaults**: Automatically selects optimal settings based on camera type
- **Real-time Preview**: Live camera preview with settings adjustment

### 2. Enhanced Live Camera Annotator
- **Camera Selection**: Choose from available camera devices
- **Resolution Control**: Select from 1080p, 720p, or 480p
- **Frame Rate Control**: Choose from 60, 30, or 24 FPS
- **BRIO Optimization**: Automatic optimization for BRIO cameras
- **Quality Indicators**: Visual indicators for camera status and optimization

### 3. Camera Settings Page
- **Comprehensive Configuration**: Advanced camera settings and optimization
- **BRIO-Specific Tips**: Special recommendations for BRIO camera users
- **Settings Persistence**: Save and restore camera configurations
- **Advanced Controls**: Fine-tune exposure, focus, and image processing

### 4. Camera Test Utility
- **Quality Analysis**: Test camera functionality and image quality
- **Brightness/Contrast Analysis**: Automatic analysis of lighting conditions
- **Resolution Verification**: Verify actual vs. expected resolution
- **Test Recording**: Record test videos for quality verification
- **Recommendations**: Automatic recommendations for optimal settings

## 🚀 Getting Started

### For Data Analysts

1. **Access Camera Settings**:
   - Navigate to "Camera Settings" in the main navigation
   - Or use the camera selector in the Live Camera section

2. **Select Your Camera**:
   - The system will automatically detect available cameras
   - BRIO cameras will be automatically identified and optimized
   - Choose your preferred camera from the dropdown

3. **Configure Settings**:
   - Adjust resolution and frame rate as needed
   - For BRIO cameras, recommended settings are pre-selected
   - Fine-tune advanced settings if required

4. **Test Your Setup**:
   - Use the Camera Test utility to verify functionality
   - Check image quality and lighting conditions
   - Record test videos to ensure optimal quality

## 📋 BRIO Camera Optimization

### Automatic Optimizations
- **Resolution**: 1920x1080 (1080p)
- **Frame Rate**: 60 FPS
- **Quality**: High
- **Auto Exposure**: Enabled
- **Auto Focus**: Enabled
- **Auto White Balance**: Enabled
- **Noise Reduction**: Enabled
- **Stabilization**: Disabled (for accurate hand tracking)

### Manual Adjustments
If you need to adjust BRIO settings manually:

1. Go to Camera Settings page
2. Select your BRIO camera
3. Modify settings as needed
4. Use the Camera Test utility to verify changes
5. Save your configuration

## 🛠️ Technical Details

### Camera Detection
The system uses the WebRTC `getUserMedia` API to:
- Enumerate available camera devices
- Detect camera types by name (BRIO, OAK, etc.)
- Request appropriate permissions
- Handle device switching

### Settings Management
- Settings are stored in localStorage
- Can be extended to save to backend
- Automatic optimization based on camera type
- Fallback to default settings for unknown cameras

### Quality Analysis
The Camera Test utility analyzes:
- Actual vs. expected resolution
- Image brightness levels
- Contrast ratios
- Frame capture quality
- Recording functionality

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Detected**:
   - Check browser permissions
   - Ensure camera is not in use by another application
   - Try refreshing the page

2. **Poor Image Quality**:
   - Use the Camera Test utility to analyze lighting
   - Adjust camera settings in the Camera Settings page
   - Check for adequate lighting conditions

3. **BRIO Not Optimized**:
   - Verify camera name contains "BRIO"
   - Check camera settings page for optimization status
   - Manually select BRIO-optimized settings

4. **Recording Issues**:
   - Ensure camera is started before recording
   - Check browser supports WebM recording
   - Verify sufficient storage space

### Browser Compatibility
- **Chrome**: Full support for all features
- **Firefox**: Full support for all features
- **Safari**: Limited support (may need adjustments)
- **Edge**: Full support for all features

## 📝 Usage Examples

### Basic Camera Selection
```javascript
// Camera selector automatically detects and optimizes
<CameraSelector 
  onCameraSelect={setSelectedCamera}
  onSettingsChange={setCameraSettings}
  showSettings={true}
/>
```

### BRIO-Specific Configuration
```javascript
// BRIO cameras get automatic optimization
if (camera.isBRIO) {
  setCameraSettings({
    resolution: '1920x1080',
    frameRate: 60,
    quality: 'high'
  });
}
```

### Camera Testing
```javascript
// Test camera functionality and quality
<CameraTest 
  selectedCamera={selectedCamera}
  cameraSettings={cameraSettings}
/>
```

## 🎯 Best Practices

### For Data Analysts
1. **Always test your camera setup** before recording important data
2. **Use BRIO cameras** when available for best quality
3. **Maintain good lighting** for optimal hand tracking
4. **Save your settings** for consistent recording quality
5. **Regularly test** camera functionality

### For Developers
1. **Handle camera permissions** gracefully
2. **Provide fallback options** for unsupported features
3. **Optimize for BRIO cameras** when detected
4. **Save user preferences** for better UX
5. **Test across different browsers** and devices

## 🔄 Future Enhancements

Planned improvements include:
- **Backend Settings Storage**: Save settings to user accounts
- **Camera Profiles**: Pre-configured settings for different use cases
- **Advanced Analytics**: More detailed quality analysis
- **Multi-Camera Recording**: Support for multiple simultaneous cameras
- **AI-Powered Optimization**: Automatic settings based on environment

---

For technical support or feature requests, please contact the development team. 