import React, { useState, useEffect, useRef } from 'react';
import { Camera, Video, Settings, CheckCircle, AlertCircle, RefreshCw, Zap, Brain } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function CameraSelector({ 
  onCameraSelect, 
  onSettingsChange, 
  showSettings = true,
  autoStart = false 
}) {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [stream, setStream] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cameraSettings, setCameraSettings] = useState({
    resolution: '1920x1080',
    frameRate: 60,
    quality: 'high'
  });
  const videoRef = useRef(null);

  // Camera detection patterns
  const cameraPatterns = {
    brio: /brio|logitech.*brio/i,
    oak: /oak|depthai|luxonis/i,
    intel: /intel.*realsense|realsense/i,
    kinect: /kinect|microsoft.*kinect/i,
    iphone: /iphone|ipad|apple.*camera/i,
    android: /android|camera.*app|mobile.*camera/i,
    webcam: /webcam|camera|usb.*camera/i
  };

  // Detect camera type based on device name and user agent
  const detectCameraType = (deviceLabel) => {
    const userAgent = navigator.userAgent;
    
    // Check for mobile devices first
    if (/iPhone|iPad|iPod/i.test(userAgent)) {
      return 'iphone';
    }
    if (/Android/i.test(userAgent)) {
      return 'android';
    }
    
    // Check device label patterns
    for (const [type, pattern] of Object.entries(cameraPatterns)) {
      if (pattern.test(deviceLabel)) {
        return type;
      }
    }
    
    // Default based on device type
    if (isMobileDevice()) {
      return 'android'; // Default mobile camera
    }
    
    return 'webcam';
  };

  // Detect if device is mobile
  const isMobileDevice = () => {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  };

  // Get optimal settings for camera type
  const getOptimalSettings = (cameraType) => {
    const isMobile = isMobileDevice();
    
    const settings = {
      brio: {
        resolution: '1920x1080',
        frameRate: 60,
        quality: 'high',
        constraints: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 60 }
        }
      },
      oak: {
        resolution: '1920x1080',
        frameRate: 30,
        quality: 'high',
        constraints: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        }
      },
      intel: {
        resolution: '1280x720',
        frameRate: 30,
        quality: 'high',
        constraints: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      },
      kinect: {
        resolution: '1280x720',
        frameRate: 30,
        quality: 'medium',
        constraints: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      },
      iphone: {
        resolution: isMobile ? '1920x1080' : '1280x720',
        frameRate: isMobile ? 30 : 30,
        quality: 'high',
        constraints: {
          width: { ideal: isMobile ? 1920 : 1280 },
          height: { ideal: isMobile ? 1080 : 720 },
          frameRate: { ideal: 30 },
          facingMode: { ideal: 'user' }
        }
      },
      android: {
        resolution: isMobile ? '1920x1080' : '1280x720',
        frameRate: isMobile ? 30 : 30,
        quality: 'high',
        constraints: {
          width: { ideal: isMobile ? 1920 : 1280 },
          height: { ideal: isMobile ? 1080 : 720 },
          frameRate: { ideal: 30 },
          facingMode: { ideal: 'user' }
        }
      },
      webcam: {
        resolution: isMobile ? '1280x720' : '1280x720',
        frameRate: 30,
        quality: 'medium',
        constraints: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      }
    };
    return settings[cameraType] || settings.webcam;
  };

  // Load available cameras
  const loadCameras = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access not supported in this browser');
      }

      // Get all media devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');

      if (videoDevices.length === 0) {
        throw new Error('No cameras found');
      }

      // Process camera information
      const cameraList = videoDevices.map(device => {
        const cameraType = detectCameraType(device.label);
        const optimalSettings = getOptimalSettings(cameraType);
        
        return {
          deviceId: device.deviceId,
          label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
          type: cameraType,
          isBRIO: cameraType === 'brio',
          isOAK: cameraType === 'oak',
          isIntel: cameraType === 'intel',
          isKinect: cameraType === 'kinect',
          optimalSettings,
          capabilities: {
            maxWidth: 1920,
            maxHeight: 1080,
            maxFrameRate: cameraType === 'brio' ? 60 : 30
          }
        };
      });

      setCameras(cameraList);

      // Auto-select first camera if none selected
      if (!selectedCamera && cameraList.length > 0) {
        const firstCamera = cameraList[0];
        setSelectedCamera(firstCamera);
        setCameraSettings(firstCamera.optimalSettings);
        
        if (onCameraSelect) {
          onCameraSelect(firstCamera);
        }
        if (onSettingsChange) {
          onSettingsChange(firstCamera.optimalSettings);
        }
      }

    } catch (err) {
      console.error('Error loading cameras:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Start camera stream
  const startCamera = async (camera) => {
    try {
      setIsLoading(true);
      setError(null);

      // Stop existing stream
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }

      // Get optimal constraints for camera type
      const optimalSettings = getOptimalSettings(camera.type);
      const constraints = {
        video: {
          deviceId: { exact: camera.deviceId },
          ...optimalSettings.constraints
        },
        audio: false
      };

      // Request camera access
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);

      // Set video source
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      // Update settings
      setCameraSettings(optimalSettings);
      if (onSettingsChange) {
        onSettingsChange(optimalSettings);
      }

    } catch (err) {
      console.error('Error starting camera:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Stop camera stream
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  // Handle camera selection
  const handleCameraSelect = (deviceId) => {
    const camera = cameras.find(cam => cam.deviceId === deviceId);
    if (camera) {
      setSelectedCamera(camera);
      if (onCameraSelect) {
        onCameraSelect(camera);
      }
      // Auto-start camera if autoStart is enabled
      if (autoStart) {
        startCamera(camera);
      }
    }
  };

  // Handle settings change
  const handleSettingsChange = (newSettings) => {
    setCameraSettings(newSettings);
    if (onSettingsChange) {
      onSettingsChange(newSettings);
    }
  };

  // Load cameras on component mount
  useEffect(() => {
    loadCameras();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Get camera type icon
  const getCameraIcon = (cameraType) => {
    switch (cameraType) {
      case 'brio':
        return <Camera className="h-4 w-4 text-blue-600" />;
      case 'oak':
        return <Brain className="h-4 w-4 text-purple-600" />;
      case 'intel':
        return <Zap className="h-4 w-4 text-green-600" />;
      case 'kinect':
        return <Video className="h-4 w-4 text-orange-600" />;
      case 'iphone':
        return <Camera className="h-4 w-4 text-gray-800" />;
      case 'android':
        return <Camera className="h-4 w-4 text-green-500" />;
      default:
        return <Camera className="h-4 w-4 text-gray-600" />;
    }
  };

  // Get camera type badge
  const getCameraBadge = (cameraType) => {
    const badges = {
      brio: { text: 'BRIO', className: 'bg-blue-100 text-blue-800' },
      oak: { text: 'OAK-D', className: 'bg-purple-100 text-purple-800' },
      intel: { text: 'Intel', className: 'bg-green-100 text-green-800' },
      kinect: { text: 'Kinect', className: 'bg-orange-100 text-orange-800' },
      iphone: { text: 'iPhone', className: 'bg-gray-100 text-gray-800' },
      android: { text: 'Android', className: 'bg-green-100 text-green-800' },
      webcam: { text: 'Webcam', className: 'bg-gray-100 text-gray-800' }
    };
    return badges[cameraType] || badges.webcam;
  };

  return (
    <div className="space-y-4">
      {/* Camera Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Camera Selection
            {isLoading && <RefreshCw className="h-4 w-4 animate-spin" />}
          </CardTitle>
          <CardDescription>
            Select and configure your camera for optimal sign language recording
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {cameras.length > 0 ? (
            <div className="space-y-3">
              <Select value={selectedCamera?.deviceId || ''} onValueChange={handleCameraSelect}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a camera" />
                </SelectTrigger>
                <SelectContent>
                  {cameras.map((camera) => {
                    const badge = getCameraBadge(camera.type);
                    return (
                      <SelectItem key={camera.deviceId} value={camera.deviceId}>
                        <div className="flex items-center gap-2">
                          {getCameraIcon(camera.type)}
                          <span>{camera.label}</span>
                          <Badge className={badge.className}>{badge.text}</Badge>
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>

              {selectedCamera && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Camera Information</h4>
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><strong>Type:</strong> {selectedCamera.type.toUpperCase()}</p>
                      <p><strong>Device ID:</strong> {selectedCamera.deviceId.slice(0, 8)}...</p>
                      <p><strong>Max Resolution:</strong> {selectedCamera.capabilities.maxWidth}x{selectedCamera.capabilities.maxHeight}</p>
                      <p><strong>Max Frame Rate:</strong> {selectedCamera.capabilities.maxFrameRate} FPS</p>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Current Settings</h4>
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><strong>Resolution:</strong> {cameraSettings.resolution}</p>
                      <p><strong>Frame Rate:</strong> {cameraSettings.frameRate} FPS</p>
                      <p><strong>Quality:</strong> {cameraSettings.quality}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Camera Controls */}
              <div className="flex gap-2">
                <Button 
                  onClick={() => startCamera(selectedCamera)} 
                  disabled={!selectedCamera || isLoading}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  <Video className="h-4 w-4 mr-2" />
                  {stream ? 'Restart Camera' : 'Start Camera'}
                </Button>
                
                {stream && (
                  <Button onClick={stopCamera} variant="outline">
                    Stop Camera
                  </Button>
                )}
                
                <Button onClick={loadCameras} variant="outline" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <Camera className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-4">No cameras detected</p>
              <Button onClick={loadCameras} variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Camera Preview */}
      {stream && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="h-5 w-5" />
              Camera Preview
            </CardTitle>
            <CardDescription>
              Live preview from {selectedCamera?.label}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full max-w-md mx-auto rounded-lg border"
                style={{ maxHeight: '300px' }}
              />
              {selectedCamera?.isOAK && (
                <div className="absolute top-2 right-2">
                  <Badge className="bg-purple-100 text-purple-800">
                    <Brain className="h-3 w-3 mr-1" />
                    OAK-D AI
                  </Badge>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Camera-specific Tips */}
      {selectedCamera && (
        <Card className={
          selectedCamera.isOAK ? 'border-purple-200 bg-purple-50' : 
          selectedCamera.isBRIO ? 'border-blue-200 bg-blue-50' : 
          selectedCamera.type === 'iphone' ? 'border-gray-200 bg-gray-50' :
          selectedCamera.type === 'android' ? 'border-green-200 bg-green-50' : ''
        }>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5" />
              {selectedCamera.isOAK ? 'OAK-D Optimization Tips' : 
               selectedCamera.isBRIO ? 'BRIO Optimization Tips' : 
               selectedCamera.type === 'iphone' ? 'iPhone Camera Tips' :
               selectedCamera.type === 'android' ? 'Android Camera Tips' :
               'Camera Tips'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              {selectedCamera.isOAK ? (
                <>
                  <p>• OAK-D cameras provide depth sensing and AI processing capabilities</p>
                  <p>• Optimal for 1080p at 30fps for AI analysis</p>
                  <p>• Depth data enhances hand tracking accuracy</p>
                  <p>• Ensure adequate lighting for depth sensor operation</p>
                </>
              ) : selectedCamera.isBRIO ? (
                <>
                  <p>• BRIO cameras are optimized for 1080p at 60fps</p>
                  <p>• Auto exposure and focus work best for sign language recording</p>
                  <p>• Noise reduction is recommended for clean hand tracking</p>
                  <p>• Disable stabilization for more accurate hand movement capture</p>
                </>
              ) : selectedCamera.type === 'iphone' ? (
                <>
                  <p>• iPhone cameras provide excellent quality for sign language recording</p>
                  <p>• Use front-facing camera for self-recording or back camera for others</p>
                  <p>• Ensure good lighting - iPhone cameras work well in various conditions</p>
                  <p>• Hold device steady or use a tripod for best results</p>
                  <p>• Use landscape orientation for better hand visibility</p>
                </>
              ) : selectedCamera.type === 'android' ? (
                <>
                  <p>• Android cameras offer good quality for sign language recording</p>
                  <p>• Use front-facing camera for self-recording or back camera for others</p>
                  <p>• Ensure good lighting for optimal video quality</p>
                  <p>• Hold device steady or use a tripod for best results</p>
                  <p>• Use landscape orientation for better hand visibility</p>
                </>
              ) : (
                <>
                  <p>• Ensure good lighting for optimal video quality</p>
                  <p>• Position camera at eye level for best angle</p>
                  <p>• Use 720p or higher resolution when possible</p>
                  <p>• Test different frame rates for your use case</p>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
