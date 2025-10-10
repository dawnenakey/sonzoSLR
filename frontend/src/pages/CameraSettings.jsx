import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Camera, Video, Settings, CheckCircle, AlertCircle, ArrowLeft, Save, RotateCcw, Brain, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import CameraSelector from '@/components/CameraSelector';
import CameraTest from './CameraTest';


export default function CameraSettings() {
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [cameraSettings, setCameraSettings] = useState({
    resolution: '1920x1080',
    frameRate: 60,
    quality: 'high'
  });
  const [advancedSettings, setAdvancedSettings] = useState({
    autoExposure: true,
    autoWhiteBalance: true,
    autoFocus: true,
    lowLightCompensation: false,
    noiseReduction: true,
    stabilization: false
  });

  const handleCameraSelect = (camera) => {
    setSelectedCamera(camera);
    
    // Auto-optimize settings based on camera type
    if (camera?.isBRIO) {
      setCameraSettings({
        resolution: '1920x1080',
        frameRate: 60,
        quality: 'high'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: false,
        noiseReduction: true,
        stabilization: false
      });
    } else if (camera?.isOAK) {
      setCameraSettings({
        resolution: '1920x1080',
        frameRate: 30,
        quality: 'high'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: true,
        noiseReduction: true,
        stabilization: false
      });
    } else if (camera?.type === 'iphone' || camera?.type === 'android') {
      setCameraSettings({
        resolution: '1920x1080',
        frameRate: 30,
        quality: 'high'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: true,
        noiseReduction: true,
        stabilization: true
      });
    } else {
      setCameraSettings({
        resolution: '1280x720',
        frameRate: 30,
        quality: 'medium'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: true,
        noiseReduction: true,
        stabilization: true
      });
    }
  };

  const handleSettingsChange = (settings) => {
    setCameraSettings(settings);
  };

  const handleAdvancedSettingChange = (key, value) => {
    setAdvancedSettings(prev => ({ ...prev, [key]: value }));
  };

  const resetToDefaults = () => {
    if (selectedCamera?.isBRIO) {
      setCameraSettings({
        resolution: '1920x1080',
        frameRate: 60,
        quality: 'high'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: false,
        noiseReduction: true,
        stabilization: false
      });
    } else if (selectedCamera?.isOAK) {
      setCameraSettings({
        resolution: '1920x1080',
        frameRate: 30,
        quality: 'high'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: true,
        noiseReduction: true,
        stabilization: false
      });
    } else if (selectedCamera?.type === 'iphone' || selectedCamera?.type === 'android') {
      setCameraSettings({
        resolution: '1920x1080',
        frameRate: 30,
        quality: 'high'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: true,
        noiseReduction: true,
        stabilization: true
      });
    } else {
      setCameraSettings({
        resolution: '1280x720',
        frameRate: 30,
        quality: 'medium'
      });
      setAdvancedSettings({
        autoExposure: true,
        autoWhiteBalance: true,
        autoFocus: true,
        lowLightCompensation: true,
        noiseReduction: true,
        stabilization: true
      });
    }
  };

  const saveSettings = () => {
    // Save settings to localStorage or backend
    const settings = {
      camera: selectedCamera,
      basic: cameraSettings,
      advanced: advancedSettings,
      timestamp: new Date().toISOString()
    };
    localStorage.setItem('cameraSettings', JSON.stringify(settings));
    // You could also save to backend here
    console.log('Settings saved:', settings);
  };

  return (
    <>

      <div className="max-w-7xl mx-auto p-4 sm:p-6">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6">
          <Link to="/Home" className="flex items-center gap-2 text-gray-600 hover:text-gray-900">
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Link>
          <div className="flex-1">
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <Camera className="h-8 w-8 text-blue-600" />
              Camera Settings
              {selectedCamera?.isBRIO && (
                <Badge className="bg-blue-100 text-blue-800 text-sm px-3 py-1">
                  BRIO Optimized
                </Badge>
              )}
            </h1>
            <p className="text-gray-600 mt-1">
              Configure your camera for optimal sign language recording
            </p>
          </div>
        </div>

        {/* Camera Selector */}
        <div className="mb-6">
          <CameraSelector 
            onCameraSelect={handleCameraSelect}
            onSettingsChange={handleSettingsChange}
            showSettings={true}
            autoStart={false}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Advanced Settings */}
          <div className="lg:col-span-2 space-y-6">
            {/* Current Camera Info */}
            {selectedCamera && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Video className="h-5 w-5" />
                    Current Camera: {selectedCamera.label}
                  </CardTitle>
                  <CardDescription>
                    {selectedCamera.isBRIO 
                      ? "BRIO camera detected - optimized for high-quality recording"
                      : selectedCamera.isOAK 
                      ? "OAK-D camera detected - optimized for AI processing and depth sensing"
                      : selectedCamera.isIntel
                      ? "Intel RealSense camera detected - optimized for depth and AI processing"
                      : selectedCamera.isKinect
                      ? "Kinect camera detected - optimized for motion tracking"
                      : "Standard camera - using default settings"
                    }
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Resolution:</span> {cameraSettings.resolution}
                    </div>
                    <div>
                      <span className="font-medium">Frame Rate:</span> {cameraSettings.frameRate} FPS
                    </div>
                    <div>
                      <span className="font-medium">Quality:</span> {cameraSettings.quality}
                    </div>
                    <div>
                      <span className="font-medium">Device ID:</span> 
                      <span className="text-xs text-gray-500 ml-1">{selectedCamera.deviceId.slice(0, 8)}...</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Advanced Camera Settings */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Advanced Camera Settings
                </CardTitle>
                <CardDescription>
                  Fine-tune camera parameters for optimal recording quality
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <h4 className="font-medium text-sm">Exposure & Focus</h4>
                    <div className="space-y-2">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={advancedSettings.autoExposure}
                          onChange={(e) => handleAdvancedSettingChange('autoExposure', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Auto Exposure</span>
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={advancedSettings.autoFocus}
                          onChange={(e) => handleAdvancedSettingChange('autoFocus', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Auto Focus</span>
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={advancedSettings.autoWhiteBalance}
                          onChange={(e) => handleAdvancedSettingChange('autoWhiteBalance', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Auto White Balance</span>
                      </label>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <h4 className="font-medium text-sm">Image Processing</h4>
                    <div className="space-y-2">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={advancedSettings.noiseReduction}
                          onChange={(e) => handleAdvancedSettingChange('noiseReduction', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Noise Reduction</span>
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={advancedSettings.lowLightCompensation}
                          onChange={(e) => handleAdvancedSettingChange('lowLightCompensation', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Low Light Compensation</span>
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={advancedSettings.stabilization}
                          onChange={(e) => handleAdvancedSettingChange('stabilization', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Image Stabilization</span>
                      </label>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Camera-specific Optimization Tips */}
            {selectedCamera?.isBRIO && (
              <Card className="border-blue-200 bg-blue-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-blue-800">
                    <CheckCircle className="h-5 w-5" />
                    BRIO Optimization Tips
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm text-blue-700">
                    <p>• BRIO cameras are optimized for 1080p at 60fps</p>
                    <p>• Auto exposure and focus work best for sign language recording</p>
                    <p>• Noise reduction is recommended for clean hand tracking</p>
                    <p>• Disable stabilization for more accurate hand movement capture</p>
                  </div>
                </CardContent>
              </Card>
            )}

            {selectedCamera?.isOAK && (
              <Card className="border-purple-200 bg-purple-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-purple-800">
                    <Brain className="h-5 w-5" />
                    OAK-D Optimization Tips
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm text-purple-700">
                    <p>• OAK-D cameras provide depth sensing and AI processing capabilities</p>
                    <p>• Optimal for 1080p at 30fps for AI analysis and depth processing</p>
                    <p>• Depth data enhances hand tracking accuracy significantly</p>
                    <p>• Ensure adequate lighting for depth sensor operation</p>
                    <p>• Low light compensation helps with depth sensor performance</p>
                    <p>• Disable stabilization to preserve depth data accuracy</p>
                  </div>
                </CardContent>
              </Card>
            )}

            {selectedCamera?.isIntel && (
              <Card className="border-green-200 bg-green-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-green-800">
                    <Zap className="h-5 w-5" />
                    Intel RealSense Optimization Tips
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm text-green-700">
                    <p>• Intel RealSense cameras provide depth and RGB data</p>
                    <p>• Optimal for 720p at 30fps for balanced performance</p>
                    <p>• Depth data improves hand tracking accuracy</p>
                    <p>• Ensure good lighting for both RGB and depth sensors</p>
                    <p>• Use low light compensation for better depth sensing</p>
                  </div>
                </CardContent>
              </Card>
            )}

            {(selectedCamera?.type === 'iphone' || selectedCamera?.type === 'android') && (
              <Card className="border-gray-200 bg-gray-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-gray-800">
                    <Camera className="h-5 w-5" />
                    Mobile Camera Optimization Tips
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm text-gray-700">
                    <p>• Mobile cameras provide excellent quality for sign language recording</p>
                    <p>• Use front-facing camera for self-recording or back camera for others</p>
                    <p>• Ensure good lighting - mobile cameras work well in various conditions</p>
                    <p>• Hold device steady or use a tripod for best results</p>
                    <p>• Use landscape orientation for better hand visibility</p>
                    <p>• Enable stabilization for smoother video recording</p>
                    <p>• Keep hands within frame and well-lit for best recognition</p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3">
              <Button onClick={saveSettings} className="bg-blue-600 hover:bg-blue-700">
                <Save className="mr-2 h-4 w-4" />
                Save Settings
              </Button>
              <Button onClick={resetToDefaults} variant="outline">
                <RotateCcw className="mr-2 h-4 w-4" />
                Reset to Defaults
              </Button>
            </div>
          </div>
        </div>

        {/* Camera Test Section */}
        <div className="mt-8">
          <CameraTest 
            selectedCamera={selectedCamera}
            cameraSettings={cameraSettings}
          />
        </div>
      </div>
    </>
  );
} 