import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Camera, Video, CheckCircle, AlertCircle, Play, Square, Download } from 'lucide-react';

export default function CameraTest({ selectedCamera, cameraSettings }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isTesting, setIsTesting] = useState(false);
  const [testResults, setTestResults] = useState(null);
  const [recording, setRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [stream, setStream] = useState(null);

  const runCameraTest = async () => {
    if (!selectedCamera) {
      alert('Please select a camera first');
      return;
    }

    setIsTesting(true);
    setTestResults(null);

    try {
      // Start camera stream
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: selectedCamera.deviceId ? { exact: selectedCamera.deviceId } : undefined,
          width: { ideal: parseInt(cameraSettings.resolution.split('x')[0]) },
          height: { ideal: parseInt(cameraSettings.resolution.split('x')[1]) },
          frameRate: { ideal: cameraSettings.frameRate }
        },
        audio: true
      });

      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      // Wait for video to load
      await new Promise(resolve => {
        if (videoRef.current) {
          videoRef.current.onloadedmetadata = resolve;
        }
      });

      // Capture test frame
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        // Analyze frame quality
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Calculate brightness
        let brightness = 0;
        for (let i = 0; i < data.length; i += 4) {
          brightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
        }
        brightness = brightness / (data.length / 4);

        // Calculate contrast (simplified)
        let contrast = 0;
        for (let i = 0; i < data.length; i += 4) {
          const pixelBrightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
          contrast += Math.abs(pixelBrightness - brightness);
        }
        contrast = contrast / (data.length / 4);

        // Get actual resolution and frame rate
        const actualResolution = `${video.videoWidth}x${video.videoHeight}`;
        const expectedResolution = cameraSettings.resolution;

        const results = {
          cameraName: selectedCamera.label,
          isBRIO: selectedCamera.isBRIO,
          actualResolution,
          expectedResolution,
          resolutionMatch: actualResolution === expectedResolution,
          brightness: Math.round(brightness),
          contrast: Math.round(contrast),
          brightnessStatus: brightness > 50 && brightness < 200 ? 'good' : 'poor',
          contrastStatus: contrast > 30 ? 'good' : 'poor',
          timestamp: new Date().toISOString()
        };

        setTestResults(results);
      }

    } catch (error) {
      console.error('Camera test failed:', error);
      setTestResults({
        error: error.message,
        timestamp: new Date().toISOString()
      });
    } finally {
      setIsTesting(false);
    }
  };

  const startRecording = () => {
    if (!stream) {
      alert('Please run camera test first');
      return;
    }

    setRecordedChunks([]);
    const recorder = new MediaRecorder(stream, { 
      mimeType: 'video/webm;codecs=vp9' 
    });
    
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        setRecordedChunks(prev => [...prev, e.data]);
      }
    };

    recorder.onstop = () => {
      console.log('Recording stopped');
    };

    setMediaRecorder(recorder);
    recorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorder && recording) {
      mediaRecorder.stop();
      setRecording(false);
    }
  };

  const downloadTestVideo = () => {
    if (recordedChunks.length === 0) return;

    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `camera-test-${selectedCamera?.label || 'unknown'}-${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const stopTest = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setTestResults(null);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Camera className="h-5 w-5" />
          Camera Test Utility
          {selectedCamera?.isBRIO && (
            <Badge className="bg-blue-100 text-blue-800">BRIO</Badge>
          )}
        </CardTitle>
        <CardDescription>
          Test your camera functionality and verify optimal settings for sign language recording
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Camera Preview */}
        <div className="relative">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            className="w-full max-w-md rounded-lg bg-black border-2 border-gray-200" 
          />
          <canvas 
            ref={canvasRef} 
            className="hidden" 
          />
        </div>

        {/* Test Controls */}
        <div className="flex gap-2 flex-wrap">
          {!isTesting && !testResults ? (
            <Button onClick={runCameraTest} className="bg-blue-600 hover:bg-blue-700">
              <Camera className="mr-2 h-4 w-4" />
              Run Camera Test
            </Button>
          ) : (
            <>
              <Button onClick={stopTest} variant="outline">
                Stop Test
              </Button>
              {testResults && !testResults.error && (
                <>
                  {!recording ? (
                    <Button onClick={startRecording} className="bg-green-600 hover:bg-green-700">
                      <Play className="mr-2 h-4 w-4" />
                      Start Recording Test
                    </Button>
                  ) : (
                    <Button onClick={stopRecording} className="bg-red-600 hover:bg-red-700">
                      <Square className="mr-2 h-4 w-4" />
                      Stop Recording
                    </Button>
                  )}
                  {recordedChunks.length > 0 && (
                    <Button onClick={downloadTestVideo} variant="outline">
                      <Download className="mr-2 h-4 w-4" />
                      Download Test Video
                    </Button>
                  )}
                </>
              )}
            </>
          )}
        </div>

        {/* Test Results */}
        {isTesting && (
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="text-sm text-gray-600 mt-2">Running camera test...</p>
          </div>
        )}

        {testResults && !testResults.error && (
          <div className="space-y-3">
            <h4 className="font-medium">Test Results</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Camera:</span> {testResults.cameraName}
              </div>
              <div>
                <span className="font-medium">Resolution:</span> 
                <span className={testResults.resolutionMatch ? 'text-green-600' : 'text-red-600'}>
                  {testResults.actualResolution}
                  {!testResults.resolutionMatch && ` (expected: ${testResults.expectedResolution})`}
                </span>
              </div>
              <div>
                <span className="font-medium">Brightness:</span> 
                <span className={testResults.brightnessStatus === 'good' ? 'text-green-600' : 'text-red-600'}>
                  {testResults.brightness}
                </span>
              </div>
              <div>
                <span className="font-medium">Contrast:</span> 
                <span className={testResults.contrastStatus === 'good' ? 'text-green-600' : 'text-red-600'}>
                  {testResults.contrast}
                </span>
              </div>
            </div>

            {/* Recommendations */}
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <h5 className="font-medium mb-2">Recommendations:</h5>
              <ul className="text-sm space-y-1">
                {!testResults.resolutionMatch && (
                  <li className="text-red-600">⚠️ Resolution mismatch - check camera settings</li>
                )}
                {testResults.brightnessStatus === 'poor' && (
                  <li className="text-red-600">⚠️ Brightness issues - adjust lighting or exposure</li>
                )}
                {testResults.contrastStatus === 'poor' && (
                  <li className="text-red-600">⚠️ Low contrast - improve lighting conditions</li>
                )}
                {testResults.isBRIO && testResults.resolutionMatch && testResults.brightnessStatus === 'good' && testResults.contrastStatus === 'good' && (
                  <li className="text-green-600">✓ BRIO camera optimized and ready for recording</li>
                )}
              </ul>
            </div>
          </div>
        )}

        {testResults?.error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center gap-2 text-red-600">
              <AlertCircle className="h-4 w-4" />
              <span className="font-medium">Test Failed</span>
            </div>
            <p className="text-sm text-red-600 mt-1">{testResults.error}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 