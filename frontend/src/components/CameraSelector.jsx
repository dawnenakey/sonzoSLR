import React, { useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Camera, Video, Settings, CheckCircle, AlertCircle } from 'lucide-react';

export default function CameraSelector({ 
  onCameraSelect, 
  onSettingsChange, 
  showSettings = true,
  className = "" 
}) {
  const [availableCameras, setAvailableCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [cameraSettings, setCameraSettings] = useState({
    resolution: '1920x1080',
    frameRate: 30,
    quality: 'high'
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getAvailableCameras();
  }, []);

  const getAvailableCameras = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      // Request camera permission first
      await navigator.mediaDevices.getUserMedia({ video: true });
      
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      const cameras = videoDevices.map(device => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${device.deviceId.slice(0, 8)}...`,
        isBRIO: device.label?.toLowerCase().includes('brio') || false,
        isOAK: device.label?.toLowerCase().includes('oak') || false,
        is4K: device.label?.toLowerCase().includes('4k') || 
               device.label?.toLowerCase().includes('ultra') ||
               device.label?.toLowerCase().includes('high') ||
               device.label?.toLowerCase().includes('pro'),
        isBuiltIn: device.label?.toLowerCase().includes('built-in') || 
                   device.label?.toLowerCase().includes('facetime') ||
                   device.label?.toLowerCase().includes('webcam')
      }));
      
      setAvailableCameras(cameras);
      
      // Auto-select best camera available
      const brioCamera = cameras.find(cam => cam.isBRIO);
      const oakCamera = cameras.find(cam => cam.isOAK);
      const fourKCamera = cameras.find(cam => cam.is4K);
      
      if (brioCamera) {
        setSelectedCamera(brioCamera.deviceId);
        // Optimize settings for BRIO
        setCameraSettings({
          resolution: '1920x1080',
          frameRate: 60,
          quality: 'high'
        });
      } else if (fourKCamera) {
        setSelectedCamera(fourKCamera.deviceId);
        // Optimize settings for 4K camera
        setCameraSettings({
          resolution: '3840x2160',
          frameRate: 30,
          quality: 'high'
        });
      } else if (oakCamera) {
        setSelectedCamera(oakCamera.deviceId);
        // Optimize settings for OAK
        setCameraSettings({
          resolution: '1280x720',
          frameRate: 30,
          quality: 'medium'
        });
      } else if (cameras.length > 0) {
        setSelectedCamera(cameras[0].deviceId);
      }
      
      // Notify parent component
      if (onCameraSelect && cameras.length > 0) {
        const selected = brioCamera || oakCamera || cameras[0];
        onCameraSelect(selected);
      }
      
    } catch (err) {
      console.error('Error getting cameras:', err);
      setError('Could not access camera devices. Please check permissions.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCameraChange = (deviceId) => {
    setSelectedCamera(deviceId);
    const camera = availableCameras.find(cam => cam.deviceId === deviceId);
    
    if (onCameraSelect) {
      onCameraSelect(camera);
    }
    
    // Auto-optimize settings based on camera type
    if (camera?.isBRIO) {
      const newSettings = {
        resolution: '1920x1080',
        frameRate: 60,
        quality: 'high'
      };
      setCameraSettings(newSettings);
      if (onSettingsChange) {
        onSettingsChange(newSettings);
      }
    } else if (camera?.is4K) {
      const newSettings = {
        resolution: '3840x2160',
        frameRate: 30,
        quality: 'high'
      };
      setCameraSettings(newSettings);
      if (onSettingsChange) {
        onSettingsChange(newSettings);
      }
    } else if (camera?.isOAK) {
      const newSettings = {
        resolution: '1280x720',
        frameRate: 30,
        quality: 'medium'
      };
      setCameraSettings(newSettings);
      if (onSettingsChange) {
        onSettingsChange(newSettings);
      }
    }
  };

  const handleSettingsChange = (key, value) => {
    const newSettings = { ...cameraSettings, [key]: value };
    setCameraSettings(newSettings);
    if (onSettingsChange) {
      onSettingsChange(newSettings);
    }
  };

  const selectedCameraInfo = availableCameras.find(cam => cam.deviceId === selectedCamera);

  const getCameraIcon = (camera) => {
    if (camera.isBRIO) return <Video className="h-4 w-4 text-blue-600" />;
    if (camera.isOAK) return <Camera className="h-4 w-4 text-green-600" />;
    return <Camera className="h-4 w-4 text-gray-600" />;
  };

  const getCameraBadge = (camera) => {
    if (camera.isBRIO) return <Badge className="bg-blue-100 text-blue-800">BRIO</Badge>;
    if (camera.isOAK) return <Badge className="bg-green-100 text-green-800">OAK</Badge>;
    if (camera.is4K) return <Badge className="bg-purple-100 text-purple-800">4K</Badge>;
    if (camera.isBuiltIn) return <Badge className="bg-gray-100 text-gray-800">Built-in</Badge>;
    return null;
  };

  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Camera Selection
          </CardTitle>
          <CardDescription>Loading available cameras...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse">
            <div className="h-10 bg-gray-200 rounded mb-4"></div>
            {showSettings && (
              <div className="grid grid-cols-2 gap-2">
                <div className="h-10 bg-gray-200 rounded"></div>
                <div className="h-10 bg-gray-200 rounded"></div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <AlertCircle className="h-5 w-5" />
            Camera Access Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-600 mb-4">{error}</p>
          <Button onClick={getAvailableCameras} variant="outline">
            Retry Camera Access
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Camera className="h-5 w-5" />
          Camera Selection
          {selectedCameraInfo?.isBRIO && (
            <CheckCircle className="h-5 w-5 text-blue-600" />
          )}
        </CardTitle>
        <CardDescription>
          {selectedCameraInfo?.isBRIO 
            ? "BRIO camera detected - optimized for high-quality recording"
            : "Select your preferred camera device"
          }
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Camera Selection */}
        <div>
          <Label htmlFor="camera-select" className="text-sm font-medium mb-2 block">
            Camera Device
          </Label>
          <Select value={selectedCamera} onValueChange={handleCameraChange}>
            <SelectTrigger>
              <SelectValue placeholder="Choose a camera" />
            </SelectTrigger>
            <SelectContent>
              {availableCameras.map((camera) => (
                <SelectItem key={camera.deviceId} value={camera.deviceId}>
                  <div className="flex items-center gap-2">
                    {getCameraIcon(camera)}
                    <span>{camera.label}</span>
                    {getCameraBadge(camera)}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Camera Settings */}
        {showSettings && (
          <div>
            <Label className="text-sm font-medium mb-2 block flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Camera Settings
            </Label>
            <div className="grid grid-cols-2 gap-2">
              <Select 
                value={cameraSettings.resolution} 
                onValueChange={(value) => handleSettingsChange('resolution', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="3840x2160">4K (3840x2160)</SelectItem>
                  <SelectItem value="1920x1080">1080p (1920x1080)</SelectItem>
                  <SelectItem value="1280x720">720p (1280x720)</SelectItem>
                  <SelectItem value="854x480">480p (854x480)</SelectItem>
                </SelectContent>
              </Select>
              
              <Select 
                value={cameraSettings.frameRate.toString()} 
                onValueChange={(value) => handleSettingsChange('frameRate', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="60">60 FPS (Smooth)</SelectItem>
                  <SelectItem value="30">30 FPS (Standard)</SelectItem>
                  <SelectItem value="24">24 FPS (Film)</SelectItem>
                  <SelectItem value="15">15 FPS (Low)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )}

        {/* Camera Info */}
        {selectedCameraInfo && (
          <div className="text-xs text-gray-500 pt-2 border-t">
            {selectedCameraInfo.isBRIO ? (
              <div className="flex items-center gap-2 text-blue-600">
                <CheckCircle className="h-3 w-3" />
                <span>BRIO camera detected - optimized for high-quality recording</span>
              </div>
            ) : selectedCameraInfo.isOAK ? (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle className="h-3 w-3" />
                <span>OAK camera detected - optimized for AI processing</span>
              </div>
            ) : (
              <span>Using {selectedCameraInfo.label}</span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 