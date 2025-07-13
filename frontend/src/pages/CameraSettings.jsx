import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Camera, Video, Settings, CheckCircle, AlertCircle, ArrowLeft, Save, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import CameraSelector from '../components/CameraSelector';
import CameraTest from '../components/CameraTest';
import Header from '../components/Header';

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
    // Auto-optimize settings for BRIO
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
      <Header />
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

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera Selection */}
          <div className="lg:col-span-1">
            <CameraSelector 
              onCameraSelect={handleCameraSelect}
              onSettingsChange={handleSettingsChange}
              showSettings={true}
            />
          </div>

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
                      ? "OAK camera detected - optimized for AI processing"
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

            {/* BRIO Optimization Tips */}
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