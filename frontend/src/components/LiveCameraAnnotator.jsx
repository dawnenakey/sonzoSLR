import React, { useRef, useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Camera, Video, Settings } from 'lucide-react';

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
  const [cameraSettings, setCameraSettings] = useState({
    resolution: '1920x1080',
    frameRate: 60,
    quality: 'high'
  });

  // Get available cameras on component mount
  useEffect(() => {
    getAvailableCameras();
  }, []);

  const getAvailableCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      const cameras = videoDevices.map(device => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${device.deviceId.slice(0, 8)}...`,
        isBRIO: device.label?.toLowerCase().includes('brio') || false
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
    } catch (err) {
      setError('Could not access camera: ' + err.message);
      setIsCameraActive(false);
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

  // Upload video to AWS
  const uploadVideo = async () => {
    if (recordedChunks.length === 0) return;
    setIsUploading(true);
    setError('');
    try {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const file = new File([blob], `live-recording-${Date.now()}.webm`, { type: 'video/webm' });
      // Use your existing upload API (e.g., sessionAPI.uploadVideo or similar)
      // For demo, we'll just call onVideoUploaded with the file
      if (onVideoUploaded) {
        await onVideoUploaded(file);
      }
      setRecordedChunks([]);
    } catch (err) {
      setError('Upload failed: ' + err.message);
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

  const selectedCameraInfo = availableCameras.find(cam => cam.deviceId === selectedCamera);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 flex flex-col items-center">
      <div className="flex items-center gap-2 mb-4">
        <Camera className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-bold">Live Camera Annotator</h2>
        {selectedCameraInfo?.isBRIO && (
          <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full font-medium">
            BRIO
          </span>
        )}
      </div>
      
      {error && <div className="text-red-500 mb-4 text-center">{error}</div>}
      
      {/* Camera Selection */}
      <div className="w-full max-w-md mb-4">
        <Label htmlFor="camera-select" className="text-sm font-medium mb-2 block">
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
                  {camera.label}
                  {camera.isBRIO && (
                    <span className="text-xs text-blue-600 font-medium">(BRIO)</span>
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Camera Settings */}
      <div className="w-full max-w-md mb-4">
        <Label className="text-sm font-medium mb-2 block flex items-center gap-2">
          <Settings className="h-4 w-4" />
          Camera Settings
        </Label>
        <div className="grid grid-cols-2 gap-2">
          <Select 
            value={cameraSettings.resolution} 
            onValueChange={(value) => setCameraSettings(prev => ({ ...prev, resolution: value }))}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1920x1080">1080p (1920x1080)</SelectItem>
              <SelectItem value="1280x720">720p (1280x720)</SelectItem>
              <SelectItem value="854x480">480p (854x480)</SelectItem>
            </SelectContent>
          </Select>
          
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
      </div>

      {/* Video Preview */}
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        className="w-full max-w-md rounded-lg mb-4 bg-black border-2 border-gray-200" 
      />
      
      {/* Camera Controls */}
      <div className="flex gap-2 mb-4 flex-wrap justify-center">
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
          onClick={uploadVideo} 
          className="bg-indigo-600 hover:bg-indigo-700" 
          disabled={recordedChunks.length === 0 || isUploading}
        >
          {isUploading ? 'Uploading...' : 'Upload Video'}
        </Button>
      </div>
      
      {/* Recording Status */}
      {recordedChunks.length > 0 && (
        <div className="text-sm text-gray-600 mb-2 text-center">
          Recorded video ready to upload ({(recordedChunks.reduce((acc, c) => acc + c.size, 0) / 1024 / 1024).toFixed(2)} MB)
        </div>
      )}
      
      {/* Camera Info */}
      {selectedCameraInfo && (
        <div className="text-xs text-gray-500 text-center mt-2">
          {selectedCameraInfo.isBRIO ? (
            <span className="text-blue-600">✓ BRIO camera detected - optimized for high-quality recording</span>
          ) : (
            <span>Using {selectedCameraInfo.label}</span>
          )}
        </div>
      )}
    </div>
  );
} 