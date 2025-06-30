/// <reference types="vite/client" />

import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Slider } from './ui/slider';

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

const RemoteDataCollection: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  const API_BASE_URL = import.meta.env.VITE_DATA_COLLECTION_API || 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod';

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

    const description = prompt('Enter session description (optional):');
    const tagsInput = prompt('Enter tags (comma-separated, optional):');
    const tags = tagsInput ? tagsInput.split(',').map(tag => tag.trim()) : [];

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
          description: description || undefined,
          tags
        }),
      });

      const data = await response.json();
      if (data.success) {
        await loadUserSessions();
        alert('Session created successfully!');
      }
    } catch (error) {
      console.error('Error creating session:', error);
      alert('Error creating session');
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
        audio: true
      });

      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Error accessing camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startRecording = async () => {
    if (!stream || !currentSession) {
      alert('Please start camera and select a session first');
      return;
    }

    try {
      recordedChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9'
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        uploadVideo(blob);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Start recording timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error starting recording');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    }
  };

  const uploadVideo = async (videoBlob: Blob) => {
    if (!currentSession) return;

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const filename = `recording_${Date.now()}.webm`;
      
      // First, get a presigned URL for upload
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

      // Upload the file directly to S3 using the presigned URL with progress tracking
      const xhr = new XMLHttpRequest();
      
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          setUploadProgress(progress);
        }
      };

      xhr.onload = async () => {
        if (xhr.status === 200) {
            alert('Video uploaded successfully!');
            await loadSessionVideos();
            setUploadProgress(0);
        } else {
          alert('Upload failed');
        }
        setIsUploading(false);
      };

      xhr.onerror = () => {
        alert('Upload failed');
        setIsUploading(false);
        setUploadProgress(0);
      };

      xhr.open('PUT', presignedData.uploadUrl);
      xhr.setRequestHeader('Content-Type', 'video/webm');
      xhr.send(videoBlob);

    } catch (error) {
      console.error('Error uploading video:', error);
      alert('Error uploading video');
      setIsUploading(false);
      setUploadProgress(0);
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

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4">Remote Data Collection</h2>
        
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
              {/* Camera Preview */}
              <div className="space-y-4">
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
              </div>

              {/* Upload Progress */}
              <div className="space-y-4">
                {isUploading && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold mb-2">Uploading Video...</h4>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      {Math.round(uploadProgress)}% complete
                    </p>
                  </div>
                )}

                {/* Session Videos */}
                <div>
                  <h4 className="font-semibold mb-2">Session Videos</h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {videos.map((video) => (
                      <div key={video.video_id} className="p-3 bg-gray-50 rounded-lg">
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-medium">{video.filename}</p>
                            <p className="text-sm text-gray-600">
                              {formatDuration(video.duration)} • {video.resolution}
                            </p>
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
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RemoteDataCollection; 
