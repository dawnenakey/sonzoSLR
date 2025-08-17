import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Camera, Video, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

export default function CameraTest() {
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  const [cameraInfo, setCameraInfo] = useState(null);
  const [isTesting, setIsTesting] = useState(false);
  const [testResults, setTestResults] = useState({});
  const videoRef = useRef(null);

  const startCamera = async () => {
    try {
      setIsTesting(true);
      setError(null);
      
      // Request camera access
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 60 }
        },
        audio: false
      });
      
      setStream(mediaStream);
      
      // Get camera capabilities
      const videoTrack = mediaStream.getVideoTracks()[0];
      const capabilities = videoTrack.getCapabilities();
      const settings = videoTrack.getSettings();
      
      setCameraInfo({
        deviceId: videoTrack.getSettings().deviceId,
        width: settings.width,
        height: settings.height,
        frameRate: settings.frameRate,
        deviceName: videoTrack.label || 'Unknown Camera'
      });
      
      // Test camera functionality
      await testCameraFunctionality(mediaStream);
      
    } catch (err) {
      console.error('Camera error:', err);
      setError(getErrorMessage(err));
    } finally {
      setIsTesting(false);
    }
  };

  const testCameraFunctionality = async (mediaStream) => {
    const results = {};
    
    try {
      // Test 1: Camera access
      results.access = 'success';
      
      // Test 2: Video stream
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        results.videoStream = 'success';
      }
      
      // Test 3: Resolution
      const videoTrack = mediaStream.getVideoTracks()[0];
      const settings = videoTrack.getSettings();
      if (settings.width >= 1280 && settings.height >= 720) {
        results.resolution = 'success';
      } else {
        results.resolution = 'warning';
      }
      
      // Test 4: Frame rate
      if (settings.frameRate >= 30) {
        results.frameRate = 'success';
      } else {
        results.frameRate = 'warning';
      }
      
      // Test 5: Camera type detection (BRIO-like)
      const isHighQuality = settings.width >= 1920 && settings.height >= 1080 && settings.frameRate >= 30;
      results.cameraType = isHighQuality ? 'high_quality' : 'standard';
      
    } catch (err) {
      results.error = err.message;
    }
    
    setTestResults(results);
  };

  const getErrorMessage = (error) => {
    if (error.name === 'NotAllowedError') {
      return {
        title: 'Camera Permission Denied',
        description: 'Please allow camera access in your browser. Click the camera icon in the address bar and select "Allow".'
      };
    } else if (error.name === 'NotFoundError') {
      return {
        title: 'No Camera Found',
        description: 'No camera detected. Please check your camera connection and try again.'
      };
    } else if (error.name === 'NotReadableError') {
      return {
        title: 'Camera in Use',
        description: 'Your camera is being used by another application. Please close other camera apps and try again.'
      };
    } else {
      return {
        title: 'Camera Error',
        description: `Unexpected error: ${error.message}`
      };
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setCameraInfo(null);
      setTestResults({});
    }
  };

  const getTestIcon = (result) => {
    if (result === 'success') return <CheckCircle className="h-5 w-5 text-green-500" />;
    if (result === 'warning') return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
    return <XCircle className="h-5 w-5 text-red-500" />;
  };

  const getTestColor = (result) => {
    if (result === 'success') return 'text-green-600';
    if (result === 'warning') return 'text-yellow-600';
    return 'text-red-600';
  };

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Camera Test Tool</h1>
          <p className="text-lg text-gray-600">
            Test your camera for sign language annotation
          </p>
        </div>

        {/* Camera Controls */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-5 w-5" />
              Camera Controls
            </CardTitle>
            <CardDescription>
              Start your camera to test functionality and compatibility
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <Button 
                onClick={startCamera} 
                disabled={isTesting || !!stream}
                className="flex items-center gap-2"
              >
                <Video className="h-4 w-4" />
                {isTesting ? 'Starting Camera...' : 'Start Camera Test'}
              </Button>
              
              {stream && (
                <Button 
                  onClick={stopCamera} 
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <XCircle className="h-4 w-4" />
                  Stop Camera
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>{error.title}</AlertTitle>
            <AlertDescription>{error.description}</AlertDescription>
          </Alert>
        )}

        {/* Camera Info */}
        {cameraInfo && (
          <Card>
            <CardHeader>
              <CardTitle>Camera Information</CardTitle>
              <CardDescription>Details about your connected camera</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Device:</span> {cameraInfo.deviceName}
                </div>
                <div>
                  <span className="font-medium">Resolution:</span> {cameraInfo.width} × {cameraInfo.height}
                </div>
                <div>
                  <span className="font-medium">Frame Rate:</span> {cameraInfo.frameRate} FPS
                </div>
                <div>
                  <span className="font-medium">Type:</span> 
                  <span className={`ml-2 ${cameraInfo.width >= 1920 ? 'text-green-600' : 'text-blue-600'}`}>
                    {cameraInfo.width >= 1920 ? 'High Quality (BRIO-like)' : 'Standard Camera'}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Test Results */}
        {Object.keys(testResults).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Test Results</CardTitle>
              <CardDescription>Camera functionality test results</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(testResults).map(([test, result]) => (
                  <div key={test} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="font-medium capitalize">
                      {test.replace(/([A-Z])/g, ' $1').trim()}
                    </span>
                    <div className="flex items-center gap-2">
                      {getTestIcon(result)}
                      <span className={getTestColor(result)}>
                        {result === 'success' ? 'Passed' : 
                         result === 'warning' ? 'Warning' : 'Failed'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Video Preview */}
        {stream && (
          <Card>
            <CardHeader>
              <CardTitle>Camera Preview</CardTitle>
              <CardDescription>Live feed from your camera</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full max-w-2xl mx-auto rounded-lg border"
                />
                <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                  Live
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Instructions */}
        <Card>
          <CardHeader>
            <CardTitle>How to Use</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold">1</div>
              <p>Click "Start Camera Test" to begin testing your camera</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold">2</div>
              <p>Allow camera access when prompted by your browser</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold">3</div>
              <p>Review the test results and camera information</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold">4</div>
              <p>If tests pass, your camera is ready for sign language annotation!</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
