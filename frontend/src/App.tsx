import { useState, useEffect } from "react";
import "./App.css";
import { ImportVideoDialog } from "./components/ImportVideoDialog";
import { Button } from "./components/ui/button";
import AnnotationControls from "./components/AnnotationControls";
import AnnotationList from "./components/AnnotationList";
import AnnotationTimeline from "./components/AnnotationTimeline";
import VideoPlayer from "./components/VideoPlayer";
import { awsAPI } from './api/awsClient';

function App() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [annotations, setAnnotations] = useState<any[]>([]);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [currentUser, setCurrentUser] = useState<any>(null); // You may want to wire up Cognito or your own auth
  const [sessions, setSessions] = useState<any[]>([]);
  const [currentSession, setCurrentSession] = useState<any | null>(null);
  const [currentVideo, setCurrentVideo] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    // You may want to add pagination or filtering
    // This assumes an endpoint like GET /sessions
    // If not available, you can skip this or implement as needed
    // setSessions(await awsAPI.sessions.list());
  };

  const handleFileSelect = async (file: File) => {
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setSelectedFile(file);
    setAnnotations([]); // Clear old annotations
  };

  const handleReject = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setVideoUrl(null);
    setSelectedFile(null);
    setAnnotations([]);
    setCurrentVideo(null);
  };

  const handleAcceptAndUpload = async () => {
    if (!selectedFile) {
      alert("No file selected!");
      return;
    }
    setIsUploading(true);
    try {
      // Create a new session if none exists
      let session = currentSession;
      if (!session) {
        session = await awsAPI.sessions.create({ name: selectedFile.name, description: "Video uploaded from annotation tool" });
        setCurrentSession(session);
        await loadSessions();
      }
      // Upload video to the session
      const video = await awsAPI.sessions.uploadVideo(session.id, selectedFile);
      console.log('Uploaded video object:', video);
      if (!video.id) {
        alert('Upload succeeded but video.id is missing!');
      }
      setCurrentVideo(video);
      alert('File uploaded successfully! You can now start annotating.');
    } catch (error) {
      console.error("Upload failed:", error);
      alert(`Upload failed: ${error}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleCreateAnnotation = async (startTime: number, endTime: number, label: string, notes?: string) => {
    if (!currentVideo) return;
    try {
      const annotation = await awsAPI.annotations.create({
        videoId: currentVideo.id,
        startTime,
        endTime,
        label,
        confidence: 1.0,
        notes,
      });
      setAnnotations(prev => [...prev, annotation]);
    } catch (error) {
      console.error('Failed to create annotation:', error);
      alert('Failed to create annotation');
    }
  };

  const handleUpdateAnnotation = async (annotationId: string, updates: any) => {
    if (!currentVideo) return;
    try {
      const updatedAnnotation = await awsAPI.annotations.update(annotationId, updates);
      setAnnotations(prev => prev.map(a => a.id === annotationId ? updatedAnnotation : a));
    } catch (error) {
      console.error('Failed to update annotation:', error);
      alert('Failed to update annotation');
    }
  };

  const handleDeleteAnnotation = async (annotationId: string) => {
    if (!currentVideo) return;
    try {
      await awsAPI.annotations.delete(annotationId);
      setAnnotations(prev => prev.filter(a => a.id !== annotationId));
    } catch (error) {
      console.error('Failed to delete annotation:', error);
      alert('Failed to delete annotation');
    }
  };

  // Load annotations when video changes
  useEffect(() => {
    if (currentVideo) {
      awsAPI.annotations.getByVideo(currentVideo.id).then(setAnnotations);
    }
  }, [currentVideo]);

  return (
    <div className="App">
      <header className="App-header">
        <div className="flex justify-between items-center w-full max-w-7xl mx-auto px-4">
          <h1>SpokHand SLR Annotation Tool</h1>
        </div>
      </header>
      <main className="p-4 max-w-7xl mx-auto">
        {/* Session Management and Video Upload UI remains unchanged */}
        <div className="video-container mx-auto bg-black rounded-lg overflow-hidden">
          {videoUrl && <VideoPlayer src={videoUrl} onTimeUpdate={setCurrentTime} onDurationChange={setVideoDuration} />}
        </div>
        <div className="controls p-4 my-4 border rounded-md">
          {!selectedFile ? (
            <ImportVideoDialog 
              onFileSelect={handleFileSelect} 
              disabled={isUploading}
            />
          ) : (
            <div className="flex justify-center items-center gap-4">
              <p className="text-sm font-medium">
                Loaded: <strong>{selectedFile.name}</strong>
              </p>
              <Button onClick={handleReject} variant="destructive" disabled={isUploading}>
                Reject
              </Button>
              <Button onClick={handleAcceptAndUpload} disabled={isUploading}>
                {isUploading ? 'Uploading...' : 'Accept & Upload'}
              </Button>
            </div>
          )}
        </div>
        {videoUrl && (
          <div className="annotation-section grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-4">
              <AnnotationTimeline
                annotations={annotations}
                duration={videoDuration}
                currentTime={currentTime}
                onSeek={setCurrentTime}
              />
              <AnnotationControls
                isSegmenting={isSegmenting}
                onSegmentStart={() => setIsSegmenting(true)}
                currentSegmentDuration={0}
                onSegmentEnd={() => setIsSegmenting(false)}
                onCancelSegment={() => setIsSegmenting(false)}
                onCreateAnnotation={handleCreateAnnotation}
              />
            </div>
            <div className="lg:col-span-1">
              <AnnotationList
                annotations={annotations}
                onEdit={handleUpdateAnnotation}
                onDelete={handleDeleteAnnotation}
                onSelect={(annotation) => setCurrentTime(annotation.startTime)}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
