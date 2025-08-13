import React, { useRef, useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Progress } from './ui/progress';
import { Camera, Video, Settings, Upload, CheckCircle, AlertCircle, Loader2, Eye, EyeOff } from 'lucide-react';
import EnhancedVideoViewer from './EnhancedVideoViewer';

export default function LiveCameraAnnotator({ onVideoUploaded }) {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [error, setError] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [cameraStream, setCameraStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [showAdvancedFeatures, setShowAdvancedFeatures] = useState(true);
  const [processedVideoData, setProcessedVideoData] = useState(null);
  const [cameraSettings, setCameraSettings] = useState({
    resolution: '1920x1080',
    frameRate: 60,
    quality: 'high'
  });

  // Get available cameras on component mount
  useEffect(() => {
    getAvailableCameras();
    checkConnectionHealth();
  }, []);

  const checkConnectionHealth = async () => {
    try {
      // Check if we can access camera devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasVideoDevices = devices.some(device => device.kind === 'videoinput');
      
      if (!hasVideoDevices) {
        setConnectionStatus('no-devices');
        setError('No camera devices found');
        return;
      }

      // Test camera access with timeout
      try {
        const testStream = await Promise.race([
          navigator.mediaDevices.getUserMedia({ video: true }),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Camera access timeout')), 5000)
          )
        ]);
        
        testStream.getTracks().forEach(track => track.stop());
        setConnectionStatus('connected');
        setError('');
      } catch (err) {
        if (err.name === 'NotAllowedError') {
          setConnectionStatus('permission-denied');
          setError('Camera access denied. Please allow camera permissions.');
        } else if (err.message.includes('timeout')) {
          setConnectionStatus('slow');
          setError('Camera access is slow. Please check your connection.');
        } else {
          setConnectionStatus('error');
          setError('Camera access failed: ' + err.message);
        }
      }
    } catch (err) {
      setConnectionStatus('error');
      setError('Failed to check camera availability');
    }
  };

  const getAvailableCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      const cameras = videoDevices.map(device => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${device.deviceId.slice(0, 8)}...`,
        isBRIO: device.label?.toLowerCase().includes('brio') || false,
        isOAK: device.label?.toLowerCase().includes('oak') || false
      }));
      
      setAvailableCameras(cameras);
      
      // Auto-select BRIO if available, otherwise first camera
      const brioCamera = cameras.find(cam => cam.isBRIO);
      if (brioCamera) {
        setSelectedCamera(brioCamera.deviceId);
      } else if (cameras.length > 0) {
        setSelectedCamera(cameras[0].deviceId);
      }
    } catch (err) {
      console.error('Error getting cameras:', err);
      setError('Could not access camera devices');
    }
  };

  // Start camera with selected device
  const startCamera = async () => {
    setError('');
    try {
      if (cameraStream) {
        stopCamera();
      }

      const constraints = {
        video: {
          deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
          width: { ideal: parseInt(cameraSettings.resolution.split('x')[0]) },
          height: { ideal: parseInt(cameraSettings.resolution.split('x')[1]) },
          frameRate: { ideal: cameraSettings.frameRate }
        },
        audio: true
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setCameraStream(stream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      setIsCameraActive(true);
      setConnectionStatus('connected');
    } catch (err) {
      setError('Could not access camera: ' + err.message);
      setIsCameraActive(false);
      setConnectionStatus('error');
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  // Start recording
  const startRecording = () => {
    if (!cameraStream) {
      setError('Please start camera first');
      return;
    }
    
    setRecordedChunks([]);
    setRecording(true);
    const mediaRecorder = new window.MediaRecorder(cameraStream, { 
      mimeType: 'video/webm;codecs=vp9' 
    });
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        setRecordedChunks((prev) => prev.concat(e.data));
      }
    };
    mediaRecorder.start();
  };

  // Stop recording
  const stopRecording = () => {
    setRecording(false);
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
  };

  // Process recorded video
  const processRecordedVideo = async () => {
    if (recordedChunks.length === 0) {
      setError('No recorded video to process');
      return;
    }
    
    try {
      setIsUploading(true);
      setError('');
      
      // Validate recorded chunks
      if (recordedChunks.length === 0 || recordedChunks.some(chunk => chunk.size === 0)) {
        throw new Error('Invalid recording data');
      }
      
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      
      // Validate blob
      if (blob.size === 0) {
        throw new Error('Empty video data');
      }
      
      const file = new File([blob], `live-recording-${Date.now()}.webm`, { type: 'video/webm' });
      
      // Create video data object for enhanced viewer
      const videoData = {
        file: file,
        url: URL.createObjectURL(blob),
        metadata: {
          name: file.name,
          size: file.size,
          type: file.type,
          source: 'live-camera',
          timestamp: new Date().toISOString()
        }
      };
      
      setProcessedVideoData(videoData);
      
      // Notify parent component
      if (onVideoUploaded) {
        await onVideoUploaded(file);
      }
      
      setRecordedChunks([]);
    } catch (err) {
      console.error('Video processing error:', err);
      setError('Failed to process recorded video: ' + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle camera selection change
  const handleCameraChange = (deviceId) => {
    setSelectedCamera(deviceId);
    if (isCameraActive) {
      // Restart camera with new selection
      startCamera();
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'bg-green-500';
      case 'slow': return 'bg-yellow-500';
      case 'degraded': return 'bg-yellow-500';
      case 'disconnected': return 'bg-red-500';
      case 'no-devices': return 'bg-red-500';
      case 'permission-denied': return 'bg-orange-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'slow': return 'Slow Connection';
      case 'degraded': return 'Slow Connection';
      case 'disconnected': return 'Disconnected';
      case 'no-devices': return 'No Cameras';
      case 'permission-denied': return 'Permission Denied';
      case 'error': return 'Connection Error';
      default: return 'Checking...';
    }
  };

  const selectedCameraInfo = availableCameras.find(cam => cam.deviceId === selectedCamera);

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`}></div>
          <span className="text-sm text-gray-600">{getConnectionStatusText()}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAdvancedFeatures(!showAdvancedFeatures)}
            className="flex items-center space-x-2"
          >
            {showAdvancedFeatures ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            <span>{showAdvancedFeatures ? 'Hide' : 'Show'} Advanced</span>
          </Button>
        </div>
      </div>

      {/* Camera Display */}
      <div className="mb-6">
        <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          {!isCameraActive && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
              <div className="text-center text-white">
                <Camera className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <p className="text-lg font-medium">Camera Not Active</p>
                <p className="text-sm text-gray-400">Click "Start Camera" to begin</p>
              </div>
            </div>
          )}
          {recording && (
            <div className="absolute top-4 right-4">
              <div className="bg-red-600 text-white px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                REC
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Camera Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5 text-blue-600" />
            Live Camera Recording
            {selectedCameraInfo?.isBRIO && (
              <Badge className="bg-blue-100 text-blue-800">BRIO</Badge>
            )}
            {selectedCameraInfo?.isOAK && (
              <Badge className="bg-green-100 text-green-800">OAK</Badge>
            )}
          </CardTitle>
          <CardDescription>
            Record and analyze sign language videos with advanced AI features
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {/* Connection Status */}
          <div className="flex items-center gap-2 mb-4">
            <div className={`w-2 h-2 rounded-full ${getConnectionStatusColor()}`}></div>
            <span className="text-sm text-gray-600">{getConnectionStatusText()}</span>
          </div>

          {/* Camera Selection */}
          <div className="space-y-2">
            <Label htmlFor="camera-select" className="text-sm font-medium">
              Select Camera Device
            </Label>
            <Select value={selectedCamera} onValueChange={handleCameraChange}>
              <SelectTrigger>
                <SelectValue placeholder="Choose a camera" />
              </SelectTrigger>
              <SelectContent>
                {availableCameras.map((camera) => (
                  <SelectItem key={camera.deviceId} value={camera.deviceId}>
                    <div className="flex items-center gap-2">
                      {camera.isBRIO && <Video className="h-4 w-4 text-blue-600" />}
                      {camera.isOAK && <Camera className="h-4 w-4 text-green-600" />}
                      {camera.label}
                      {camera.isBRIO && (
                        <Badge className="text-xs">BRIO</Badge>
                      )}
                      {camera.isOAK && (
                        <Badge className="text-xs">OAK</Badge>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Camera Settings */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">Resolution</Label>
              <Select 
                value={cameraSettings.resolution} 
                onValueChange={(value) => setCameraSettings(prev => ({ ...prev, resolution: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1920x1080">1920x1080 (1080p)</SelectItem>
                  <SelectItem value="1280x720">1280x720 (720p)</SelectItem>
                  <SelectItem value="854x480">854x480 (480p)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label className="text-sm font-medium">Frame Rate</Label>
              <Select 
                value={cameraSettings.frameRate.toString()} 
                onValueChange={(value) => setCameraSettings(prev => ({ ...prev, frameRate: parseInt(value) }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="60">60 FPS</SelectItem>
                  <SelectItem value="30">30 FPS</SelectItem>
                  <SelectItem value="24">24 FPS</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label className="text-sm font-medium">Quality</Label>
              <Select 
                value={cameraSettings.quality} 
                onValueChange={(value) => setCameraSettings(prev => ({ ...prev, quality: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Camera Controls */}
          <div className="flex gap-2 flex-wrap justify-center">
            {!isCameraActive ? (
              <Button onClick={startCamera} className="bg-blue-600 hover:bg-blue-700">
                <Camera className="mr-2 h-4 w-4" />
                Start Camera
              </Button>
            ) : (
              <Button onClick={stopCamera} variant="outline" className="border-red-300 text-red-600 hover:bg-red-50">
                Stop Camera
              </Button>
            )}
            
            {!recording ? (
              <Button 
                onClick={startRecording} 
                className="bg-green-600 hover:bg-green-700" 
                disabled={!isCameraActive}
              >
                <Video className="mr-2 h-4 w-4" />
                Start Recording
              </Button>
            ) : (
              <Button 
                onClick={stopRecording} 
                className="bg-red-600 hover:bg-red-700"
              >
                Stop Recording
              </Button>
            )}
            
            <Button 
              onClick={processRecordedVideo} 
              className="bg-indigo-600 hover:bg-indigo-700" 
              disabled={recordedChunks.length === 0 || isUploading}
            >
              {isUploading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Process Video
                </>
              )}
            </Button>
          </div>
          
          {/* Recording Status */}
          {recordedChunks.length > 0 && (
            <div className="text-sm text-gray-600 text-center">
              <div className="flex items-center justify-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span>
                  Recorded video ready to process ({(recordedChunks.reduce((acc, c) => acc + c.size, 0) / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
            </div>
          )}
          
          {/* Camera Info */}
          {selectedCameraInfo && (
            <div className="text-xs text-gray-500 text-center">
              {selectedCameraInfo.isBRIO ? (
                <span className="text-blue-600">✓ BRIO camera detected - optimized for high-quality recording</span>
              ) : selectedCameraInfo.isOAK ? (
                <span className="text-green-600">✓ OAK camera detected - optimized for depth analysis</span>
              ) : (
                <span>Using {selectedCameraInfo.label}</span>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Enhanced Video Viewer */}
      {processedVideoData && (
        <EnhancedVideoViewer
          videoData={processedVideoData}
          showAdvancedFeatures={showAdvancedFeatures}
          onVideoProcessed={(data) => {
            console.log('Video processed:', data);
          }}
        />
      )}
    </div>
  );
} 