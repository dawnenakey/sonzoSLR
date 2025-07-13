/// <reference types="vite/client" />

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import AWSUploadNotification from './AWSUploadNotification';

interface Session {
  session_id: string;
  session_name: string;
  description?: string;
  tags: string[];
  status: string;
  video_count: number;
  total_duration: number;
}

interface Video {
  video_id: string;
  filename: string;
  duration: number;
  resolution: string;
  hand_landmarks_detected: boolean;
  processing_status: string;
  upload_timestamp: string;
}

interface UploadStatus {
  isUploading: boolean;
  progress: number;
  status: 'idle' | 'connecting' | 'uploading' | 'processing' | 'completed' | 'error';
  message: string;
  awsUrl?: string;
  error?: string;
}

const RemoteDataCollection: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({
    isUploading: false,
    progress: 0,
    status: 'idle',
    message: ''
  });
  const [showNotification, setShowNotification] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingIntervalRef = useRef<number | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  // Space bar shortcut for recording
  const recordingAreaRef = useRef<HTMLDivElement>(null);
  const handleSpacebar = useCallback((e: KeyboardEvent) => {
    if (document.activeElement && recordingAreaRef.current && recordingAreaRef.current.contains(document.activeElement)) {
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        if (stream && !isRecording) {
          startRecording();
        } else if (isRecording) {
          stopRecording();
        }
      }
    }
  }, [stream, isRecording]);

  useEffect(() => {
    window.addEventListener('keydown', handleSpacebar);
    return () => window.removeEventListener('keydown', handleSpacebar);
  }, [handleSpacebar]);

  const API_BASE_URL = (import.meta as any).env?.VITE_DATA_COLLECTION_API || 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod';

  useEffect(() => {
    // Load user sessions on component mount
    loadUserSessions();
  }, []);

  const loadUserSessions = async () => {
    try {
      const userId = localStorage.getItem('user_id') || 'default_user';
      const response = await fetch(`${API_BASE_URL}/sessions/user/${userId}`);
      const data = await response.json();
      
      if (data.success) {
        setSessions(data.sessions);
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
    }
  };

  const createNewSession = async () => {
    const sessionName = prompt('Enter session name:');
    if (!sessionName) return;

    try {
      const userId = localStorage.getItem('user_id') || 'default_user';
      const response = await fetch(`${API_BASE_URL}/sessions/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          session_name: sessionName,
          description: `Session created on ${new Date().toLocaleString()}`,
          tags: ['remote-collection']
        }),
      });

      const data = await response.json();
      if (data.success) {
        await loadUserSessions();
        setCurrentSession(data.session);
      }
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startRecording = async () => {
    if (!stream) return;

    recordedChunksRef.current = [];
    setIsRecording(true);
    setRecordingTime(0);

    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=vp9'
    });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const videoBlob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
      uploadVideo(videoBlob);
    };

    mediaRecorder.start();
    
    // Start recording timer
    recordingIntervalRef.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
  };

  const uploadVideo = async (videoBlob: Blob) => {
    if (!currentSession) return;

    setUploadStatus({
      isUploading: true,
      progress: 0,
      status: 'connecting',
      message: 'Connecting to AWS...'
    });
    setShowNotification(true);

    try {
      const filename = `recording_${Date.now()}.webm`;
      
      // Step 1: Get presigned URL
      setUploadStatus((prev: UploadStatus) => ({
        ...prev,
        status: 'connecting',
        message: 'Getting upload URL from AWS...'
      }));

      const presignedResponse = await fetch(`${API_BASE_URL}/sessions/${currentSession.session_id}/upload-video`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: filename,
          contentType: 'video/webm'
        }),
      });

      if (!presignedResponse.ok) {
        throw new Error(`Failed to get upload URL: ${presignedResponse.statusText}`);
      }

      const presignedData = await presignedResponse.json();
      if (!presignedData.success) {
        throw new Error(`Failed to get upload URL: ${presignedData.error}`);
      }

      // Step 2: Upload to S3 with progress tracking
      setUploadStatus((prev: UploadStatus) => ({
        ...prev,
        status: 'uploading',
        message: 'Uploading video to AWS S3...'
      }));

      const xhr = new XMLHttpRequest();
      
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          setUploadStatus(prev => ({
            ...prev,
            progress,
            message: `Uploading to AWS S3... ${Math.round(progress)}%`
          }));
        }
      };

      xhr.onload = async () => {
        if (xhr.status === 200) {
          // Step 3: Processing confirmation
          setUploadStatus(prev => ({
            ...prev,
            status: 'processing',
            message: 'Video uploaded successfully! Processing in AWS...'
          }));

          // Wait a moment to show processing status
          setTimeout(async () => {
            setUploadStatus({
              isUploading: false,
              progress: 100,
              status: 'completed',
              message: '✅ Successfully uploaded to AWS!',
              awsUrl: presignedData.uploadUrl
            });

            await loadSessionVideos();
            
            // Clear success message after 5 seconds
            setTimeout(() => {
              setUploadStatus({
                isUploading: false,
                progress: 0,
                status: 'idle',
                message: ''
              });
            }, 5000);
          }, 2000);
        } else {
          setUploadStatus({
            isUploading: false,
            progress: 0,
            status: 'error',
            message: '❌ Upload failed',
            error: `HTTP ${xhr.status}: ${xhr.statusText}`
          });
        }
      };

      xhr.onerror = () => {
        setUploadStatus({
          isUploading: false,
          progress: 0,
          status: 'error',
          message: '❌ Network error during upload',
          error: 'Connection failed'
        });
      };

      xhr.open('PUT', presignedData.uploadUrl);
      xhr.setRequestHeader('Content-Type', 'video/webm');
      xhr.send(videoBlob);

    } catch (error) {
      console.error('Error uploading video:', error);
      setUploadStatus({
        isUploading: false,
        progress: 0,
        status: 'error',
        message: '❌ Upload failed',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  };

  const loadSessionVideos = async () => {
    if (!currentSession) return;

    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${currentSession.session_id}/videos`);
      const data = await response.json();
      
      if (data.success) {
        setVideos(data.videos);
      }
    } catch (error) {
      console.error('Error loading videos:', error);
    }
  };

  const selectSession = async (session: Session) => {
    setCurrentSession(session);
    await loadSessionVideos();
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  const getStatusColor = (status: UploadStatus['status']) => {
    switch (status) {
      case 'connecting': return 'bg-blue-500';
      case 'uploading': return 'bg-blue-600';
      case 'processing': return 'bg-yellow-500';
      case 'completed': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6 flex">
      {/* Main Content */}
      <div className="flex-1">
        {/* AWS Upload Notification */}
        <AWSUploadNotification
          isVisible={showNotification}
          status={uploadStatus.status}
          progress={uploadStatus.progress}
          message={uploadStatus.message}
          error={uploadStatus.error}
          awsUrl={uploadStatus.awsUrl}
          onClose={() => setShowNotification(false)}
        />
        
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-4">Remote Data Collection</h2>
          
          {/* AWS Upload Status Banner */}
          {uploadStatus.status !== 'idle' && (
            <div className={`mb-6 p-4 rounded-lg border-l-4 ${getStatusColor(uploadStatus.status)} border-l-4`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(uploadStatus.status)} animate-pulse`}></div>
                  <div>
                    <h4 className="font-semibold text-white">
                      {uploadStatus.status === 'connecting' && '🔗 Connecting to AWS...'}
                      {uploadStatus.status === 'uploading' && '📤 Uploading to AWS S3...'}
                      {uploadStatus.status === 'processing' && '⚙️ Processing in AWS...'}
                      {uploadStatus.status === 'completed' && '✅ Successfully uploaded to AWS!'}
                      {uploadStatus.status === 'error' && '❌ Upload failed'}
                    </h4>
                    <p className="text-sm text-white/90">{uploadStatus.message}</p>
                    {uploadStatus.error && (
                      <p className="text-xs text-white/80 mt-1">Error: {uploadStatus.error}</p>
                    )}
                  </div>
                </div>
                
                {uploadStatus.status === 'uploading' && (
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-white/20 rounded-full h-2">
                      <div 
                        className="bg-white h-2 rounded-full transition-all duration-300"
                        style={{ width: `${uploadStatus.progress}%` }}
                      />
                    </div>
                    <span className="text-white text-sm font-medium">
                      {Math.round(uploadStatus.progress)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Session Management */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Sessions</h3>
              <Button onClick={createNewSession} className="bg-blue-600 hover:bg-blue-700">
                Create New Session
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    currentSession?.session_id === session.session_id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => selectSession(session)}
                >
                  <h4 className="font-semibold">{session.session_name}</h4>
                  {session.description && (
                    <p className="text-sm text-gray-600 mt-1">{session.description}</p>
                  )}
                  <div className="flex flex-wrap gap-1 mt-2">
                    {session.tags.map((tag, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-xs rounded">
                        {tag}
                      </span>
                    ))}
                  </div>
                  <div className="text-sm text-gray-500 mt-2">
                    {session.video_count} videos • {formatDuration(session.total_duration)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Camera and Recording Controls */}
          {currentSession && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-4">
                Recording Session: {currentSession.session_name}
              </h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Camera Preview and Controls */}
                <div className="space-y-4" ref={recordingAreaRef} tabIndex={0}>
                  <div className="relative bg-black rounded-lg overflow-hidden">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-64 object-cover"
                    />
                    {isRecording && (
                      <div className="absolute top-4 right-4 bg-red-500 text-white px-2 py-1 rounded text-sm">
                        REC {formatTime(recordingTime)}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-2">
                    {!stream ? (
                      <Button onClick={startCamera} className="bg-green-600 hover:bg-green-700">
                        Start Camera
                      </Button>
                    ) : (
                      <Button onClick={stopCamera} className="bg-gray-600 hover:bg-gray-700">
                        Stop Camera
                      </Button>
                    )}
                    {stream && !isRecording && (
                      <Button onClick={startRecording} className="bg-red-600 hover:bg-red-700">
                        Start Recording
                      </Button>
                    )}
                    {isRecording && (
                      <Button onClick={stopRecording} className="bg-red-600 hover:bg-red-700">
                        Stop Recording
                      </Button>
                    )}
                  </div>

                  {/* Info box, tooltip, and Download button for recorded video */}
                  {recordedChunksRef.current.length > 0 && !isRecording && (
                    <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-900 relative">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold">Recorded video ready to upload</span>
                        <span className="text-xs text-blue-700">({(recordedChunksRef.current.reduce((acc, c) => acc + c.size, 0) / 1024 / 1024).toFixed(2)} MB)</span>
                        <span className="ml-2 cursor-pointer" title="The video is currently stored in your browser. Click 'Upload Video' to send it to AWS and add it to your library, or 'Download' to save a local copy before uploading.">ℹ️</span>
                      </div>
                      <ul className="list-disc pl-5 mb-2">
                        <li>The video is stored in your browser until you upload or download it.</li>
                        <li>Click <b>Upload Video</b> to send it to AWS and add it to your library.</li>
                        <li>Click <b>Download</b> to save a copy to your computer before uploading.</li>
                      </ul>
                      <div className="flex gap-2">
                        <Button
                          className="bg-indigo-600 hover:bg-indigo-700"
                          onClick={() => uploadVideo(new Blob(recordedChunksRef.current, { type: 'video/webm' }))}
                          disabled={uploadStatus.isUploading}
                        >
                          {uploadStatus.isUploading ? 'Uploading...' : 'Upload Video'}
                        </Button>
                        <Button
                          className="bg-gray-200 hover:bg-gray-300 text-gray-800"
                          onClick={() => {
                            const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `recorded-video-${Date.now()}.webm`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                          }}
                        >
                          Download
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
                {/* (Sidebar is now outside this grid) */}
              </div>
            </div>
          )}
        </div>
      </div>
      {/* Sidebar: Session Videos */}
      <aside className="w-80 ml-6 bg-gray-50 rounded-lg shadow-lg p-4 h-fit sticky top-6 self-start">
        <h4 className="font-semibold mb-2">Session Videos</h4>
        <div className="space-y-2 max-h-[70vh] overflow-y-auto">
          {/* Pending Approval Section (UI only for now) */}
          {videos.filter(video => video.processing_status === 'pending').length > 0 && (
            <div className="mb-2 p-2 bg-yellow-50 border border-yellow-200 rounded">
              <span className="font-semibold text-yellow-800">Pending Approval</span>
              <ul className="mt-1 space-y-1">
                {videos.filter(video => video.processing_status === 'pending').map((video) => (
                  <li key={video.video_id} className="flex justify-between items-center text-yellow-900 text-sm">
                    <span>{video.filename}</span>
                    <span className="italic">Pending</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Approved/Other Videos */}
          {videos.filter(video => video.processing_status !== 'pending').map((video) => (
            <div key={video.video_id} className="p-3 bg-white border rounded-lg">
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-medium">{video.filename}</p>
                  <p className="text-xs text-gray-500">
                    {new Date(video.upload_timestamp).toLocaleString()}
                  </p>
                </div>
                <div className="flex flex-col items-end">
                  <span className={`px-2 py-1 text-xs rounded ${
                    video.hand_landmarks_detected
                      ? 'bg-green-100 text-green-800'
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {video.hand_landmarks_detected ? 'Hands Detected' : 'No Hands'}
                  </span>
                  <span className={`px-2 py-1 text-xs rounded mt-1 ${
                    video.processing_status === 'completed'
                      ? 'bg-green-100 text-green-800'
                      : video.processing_status === 'error'
                      ? 'bg-red-100 text-red-800'
                      : 'bg-blue-100 text-blue-800'
                  }`}>
                    {video.processing_status}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
};

export default RemoteDataCollection; 
