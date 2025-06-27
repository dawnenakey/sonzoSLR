import React, { useRef, useState } from 'react';

export default function LiveCameraAnnotator({ onVideoUploaded }) {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [error, setError] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  // Start camera
  const startCamera = async () => {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      setError('Could not access camera: ' + err.message);
    }
  };

  // Start recording
  const startRecording = () => {
    setRecordedChunks([]);
    setRecording(true);
    const stream = videoRef.current.srcObject;
    const mediaRecorder = new window.MediaRecorder(stream, { mimeType: 'video/webm' });
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

  return (
    <div className="bg-white rounded-xl shadow-lg p-4 flex flex-col items-center">
      <h2 className="text-lg font-bold mb-2">Live Camera (Brio/Oak) Annotator</h2>
      {error && <div className="text-red-500 mb-2">{error}</div>}
      <video ref={videoRef} autoPlay playsInline className="w-full max-w-md rounded mb-4 bg-black" />
      <div className="flex gap-2 mb-4">
        <button onClick={startCamera} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Start Camera</button>
        {!recording ? (
          <button onClick={startRecording} className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700" disabled={!videoRef.current || !videoRef.current.srcObject}>Start Recording</button>
        ) : (
          <button onClick={stopRecording} className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">Stop Recording</button>
        )}
        <button onClick={uploadVideo} className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700" disabled={recordedChunks.length === 0 || isUploading}>{isUploading ? 'Uploading...' : 'Upload Video'}</button>
      </div>
      {recordedChunks.length > 0 && (
        <div className="text-sm text-gray-600 mb-2">Recorded video ready to upload ({(recordedChunks.reduce((acc, c) => acc + c.size, 0) / 1024 / 1024).toFixed(2)} MB)</div>
      )}
    </div>
  );
} 