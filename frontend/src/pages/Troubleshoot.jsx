import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  Camera, 
  AlertCircle, 
  CheckCircle, 
  XCircle, 
  Info, 
  Settings, 
  RefreshCw,
  ExternalLink,
  Download
} from 'lucide-react';

export default function Troubleshoot() {
  const [selectedCamera, setSelectedCamera] = useState('brio');
  const [troubleshootingStep, setTroubleshootingStep] = useState(0);

  const cameraTypes = [
    { id: 'brio', name: 'Logitech BRIO', icon: Camera, description: 'High-quality webcam for ASL recognition' },
    { id: 'oak', name: 'OAK-D Camera', icon: Camera, description: 'Depth camera with AI processing' },
    { id: 'general', name: 'General Camera', icon: Camera, description: 'Other camera types and devices' }
  ];

  const brioSteps = [
    {
      title: 'Check Mac Privacy Settings',
      description: 'Ensure camera permissions are enabled in System Preferences',
      solution: 'Go to System Preferences > Security & Privacy > Privacy > Camera and enable permissions for your browser.',
      icon: Settings,
      difficulty: 'Easy'
    },
    {
      title: 'Verify Camera Not in Use',
      description: 'Close other applications that might be using the camera',
      solution: 'Check if Zoom, FaceTime, or other apps are using the camera. Close them completely.',
      icon: XCircle,
      difficulty: 'Easy'
    },
    {
      title: 'Try Different USB Ports',
      description: 'Connect the BRIO to different USB ports',
      solution: 'Move the camera to a different USB port, preferably USB 3.0 or higher.',
      icon: RefreshCw,
      difficulty: 'Easy'
    },
    {
      title: 'Update Camera Drivers',
      description: 'Ensure you have the latest Logitech software',
      solution: 'Download and install the latest Logitech G HUB or Logitech Gaming Software.',
      icon: Download,
      difficulty: 'Medium'
    }
  ];

  const oakSteps = [
    {
      title: 'Verify USB-C Connection',
      description: 'Ensure proper USB-C cable connection',
      solution: 'Use a high-quality USB-C cable and ensure it\'s properly connected to both the camera and your computer.',
      icon: CheckCircle,
      difficulty: 'Easy'
    },
    {
      title: 'Check DepthAI Installation',
      description: 'Verify DepthAI library is properly installed',
      solution: 'Run `pip install depthai` and test with basic examples from the DepthAI repository.',
      icon: Download,
      difficulty: 'Medium'
    },
    {
      title: 'Ensure Proper Power Supply',
      description: 'Check if the camera is receiving adequate power',
      solution: 'Some OAK-D cameras require external power. Check the power LED indicator.',
      icon: AlertCircle,
      difficulty: 'Medium'
    },
    {
      title: 'Test with DepthAI Examples',
      description: 'Run official DepthAI examples to verify functionality',
      solution: 'Clone the DepthAI examples repository and run basic camera tests.',
      icon: ExternalLink,
      difficulty: 'Hard'
    }
  ];

  const generalSteps = [
    {
      title: 'Refresh Browser Permissions',
      description: 'Clear and reset camera permissions in your browser',
      solution: 'Go to browser settings > Privacy > Site Settings > Camera and reset permissions.',
      icon: RefreshCw,
      difficulty: 'Easy'
    },
    {
      title: 'Clear Browser Cache',
      description: 'Clear browser cache and cookies',
      solution: 'Clear browsing data and restart your browser completely.',
      icon: RefreshCw,
      difficulty: 'Easy'
    },
    {
      title: 'Try Different Browsers',
      description: 'Test camera functionality in different browsers',
      solution: 'Test in Chrome, Firefox, Safari, or Edge to isolate browser-specific issues.',
      icon: ExternalLink,
      difficulty: 'Easy'
    },
    {
      title: 'Check System Camera Permissions',
      description: 'Verify system-level camera permissions',
      solution: 'Check your operating system\'s camera privacy settings and ensure they\'re enabled.',
      icon: Settings,
      difficulty: 'Medium'
    }
  ];

  const getSteps = () => {
    switch (selectedCamera) {
      case 'brio':
        return brioSteps;
      case 'oak':
        return oakSteps;
      case 'general':
        return generalSteps;
      default:
        return brioSteps;
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Easy':
        return 'bg-green-100 text-green-800';
      case 'Medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'Hard':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const steps = getSteps();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Camera Troubleshooting</h1>
          <p className="text-gray-600 mt-2">
            Step-by-step solutions for common camera issues
          </p>
        </div>
        <Button 
          onClick={() => setTroubleshootingStep(0)}
          variant="outline"
          className="flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Reset Guide
        </Button>
      </div>

      {/* Camera Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Your Camera Type</CardTitle>
          <CardDescription>
            Choose the camera you're experiencing issues with for targeted troubleshooting
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {cameraTypes.map((camera) => {
              const IconComponent = camera.icon;
              return (
                <Button
                  key={camera.id}
                  variant={selectedCamera === camera.id ? 'default' : 'outline'}
                  className="h-auto p-4 flex flex-col items-center gap-3"
                  onClick={() => setSelectedCamera(camera.id)}
                >
                  <IconComponent className="h-8 w-8" />
                  <div className="text-center">
                    <div className="font-semibold">{camera.name}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {camera.description}
                    </div>
                  </div>
                </Button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Quick Test */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Camera Test</CardTitle>
          <CardDescription>
            Test your camera functionality before troubleshooting
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Alert>
              <Info className="h-4 w-4" />
              <AlertTitle>Test Your Camera</AlertTitle>
              <AlertDescription>
                Use the Camera Test page to verify basic camera functionality before proceeding with troubleshooting steps.
              </AlertDescription>
            </Alert>
            <Button 
              onClick={() => window.location.href = '/CameraTest'}
              className="flex items-center gap-2"
            >
              <Camera className="h-4 w-4" />
              Go to Camera Test
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Troubleshooting Steps */}
      <Card>
        <CardHeader>
          <CardTitle>Troubleshooting Steps for {cameraTypes.find(c => c.id === selectedCamera)?.name}</CardTitle>
          <CardDescription>
            Follow these steps in order to resolve your camera issues
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {steps.map((step, index) => {
              const IconComponent = step.icon;
              const isActive = troubleshootingStep === index;
              const isCompleted = troubleshootingStep > index;
              
              return (
                <div key={index} className="relative">
                  {/* Step Number */}
                  <div className="flex items-center gap-4">
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold ${
                      isCompleted 
                        ? 'bg-green-500 text-white' 
                        : isActive 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gray-200 text-gray-600'
                    }`}>
                      {isCompleted ? <CheckCircle className="h-4 w-4" /> : index + 1}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-lg font-semibold">{step.title}</h3>
                        <Badge className={getDifficultyColor(step.difficulty)}>
                          {step.difficulty}
                        </Badge>
                      </div>
                      <p className="text-gray-600 mb-3">{step.description}</p>
                      
                      {isActive && (
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <div className="flex items-start gap-3">
                            <IconComponent className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                            <div>
                              <h4 className="font-medium text-blue-900 mb-2">Solution:</h4>
                              <p className="text-blue-800">{step.solution}</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Progress Line */}
                  {index < steps.length - 1 && (
                    <div className="absolute left-4 top-8 w-0.5 h-6 bg-gray-200 ml-[-1px]" />
                  )}
                </div>
              );
            })}
          </div>
          
          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8">
            <Button
              variant="outline"
              onClick={() => setTroubleshootingStep(Math.max(0, troubleshootingStep - 1))}
              disabled={troubleshootingStep === 0}
            >
              Previous Step
            </Button>
            
            <Button
              onClick={() => setTroubleshootingStep(Math.min(steps.length - 1, troubleshootingStep + 1))}
              disabled={troubleshootingStep === steps.length - 1}
            >
              Next Step
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Additional Resources */}
      <Card>
        <CardHeader>
          <CardTitle>Additional Resources</CardTitle>
          <CardDescription>
            Helpful links and documentation for advanced troubleshooting
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Camera Test Page</h3>
              <p className="text-sm text-gray-600 mb-3">
                Test your camera functionality directly in the browser
              </p>
              <Button variant="outline" size="sm" onClick={() => window.location.href = '/CameraTest'}>
                <Camera className="h-4 w-4 mr-2" />
                Test Camera
              </Button>
            </div>
            
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Camera Settings</h3>
              <p className="text-sm text-gray-600 mb-3">
                Configure camera parameters and preferences
              </p>
              <Button variant="outline" size="sm" onClick={() => window.location.href = '/CameraSettings'}>
                <Settings className="h-4 w-4 mr-2" />
                Open Settings
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
