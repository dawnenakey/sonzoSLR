import { useState, useEffect } from "react";
import "./App.css";
import { ImportVideoDialog } from "./components/ImportVideoDialog";
import { Button } from "./components/ui/button";
import AnnotationControls from "./components/AnnotationControls";
import AnnotationList from "./components/AnnotationList";
import AnnotationTimeline from "./components/AnnotationTimeline";
import VideoPlayer from "./components/VideoPlayer";
import { 
  localAnnotationClient, 
  signIn, 
  signOut, 
  getCurrentUser, 
  createSession, 
  getSessions,
  uploadVideo,
  createAnnotation,
  updateAnnotation,
  deleteAnnotation,
  getAnnotations,
  analyzeVideo,
  exportData,
  importData,
  type Annotation,
  type Video,
  type User,
  type AnnotationSession
} from './api/localAnnotationClient';

function App() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<AnnotationSession[]>([]);
  const [currentSession, setCurrentSession] = useState<AnnotationSession | null>(null);
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Initialize app
  useEffect(() => {
    const user = getCurrentUser();
    if (user) {
      setCurrentUser(user);
      loadSessions();
    } else {
      // Auto-login for demo purposes
      handleSignIn('demo@spokhand.com', 'demo123');
    }
  }, []);

  const handleSignIn = async (email: string, password: string) => {
    const result = await signIn(email, password);
    if (result.success && result.user) {
      setCurrentUser(result.user);
      loadSessions();
    }
  };

  const handleSignOut = async () => {
    await signOut();
    setCurrentUser(null);
    setSessions([]);
    setCurrentSession(null);
    setCurrentVideo(null);
  };

  const loadSessions = async () => {
    const sessionList = await getSessions();
    setSessions(sessionList);
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
    if (!selectedFile || !currentUser) {
      alert("No file selected or user not logged in!");
      return;
    }

    setIsUploading(true);
    try {
      // Create a new session if none exists
      let session = currentSession;
      if (!session) {
        session = await createSession(selectedFile.name, "Video uploaded from annotation tool");
        setCurrentSession(session);
        await loadSessions();
      }

      // Upload video to the session
      const video = await uploadVideo(session.id, selectedFile);
      setCurrentVideo(video);
      
      alert('File uploaded successfully! You can now start annotating.');
      // Don't reset UI - keep the video for annotation

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
      const annotation = await createAnnotation(currentVideo.id, {
        videoId: currentVideo.id,
        startTime,
        endTime,
        label,
        notes,
        confidence: 1.0, // Manual annotation
        handShape: 'Unknown',
        location: 'neutral space'
      });

      setAnnotations(prev => [...prev, annotation]);
    } catch (error) {
      console.error('Failed to create annotation:', error);
      alert('Failed to create annotation');
    }
  };

  const handleUpdateAnnotation = async (annotationId: string, updates: Partial<Annotation>) => {
    if (!currentVideo) return;

    try {
      const updatedAnnotation = await updateAnnotation(currentVideo.id, annotationId, updates);
      setAnnotations(prev => prev.map(a => a.id === annotationId ? updatedAnnotation : a));
    } catch (error) {
      console.error('Failed to update annotation:', error);
      alert('Failed to update annotation');
    }
  };

  const handleDeleteAnnotation = async (annotationId: string) => {
    if (!currentVideo) return;

    try {
      await deleteAnnotation(currentVideo.id, annotationId);
      setAnnotations(prev => prev.filter(a => a.id !== annotationId));
    } catch (error) {
      console.error('Failed to delete annotation:', error);
      alert('Failed to delete annotation');
    }
  };

  const handleAnalyzeVideo = async () => {
    if (!currentVideo) return;

    setIsAnalyzing(true);
    try {
      const newAnnotations = await analyzeVideo(currentVideo.id);
      setAnnotations(prev => [...prev, ...newAnnotations]);
      alert(`AI analysis complete! Found ${newAnnotations.length} annotations.`);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('AI analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExportData = async () => {
    try {
      const data = await exportData();
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `spokhand-annotations-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed');
    }
  };

  const handleImportData = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      await importData(text);
      await loadSessions();
      alert('Data imported successfully!');
    } catch (error) {
      console.error('Import failed:', error);
      alert('Import failed');
    }
  };

  // Load annotations when video changes
  useEffect(() => {
    if (currentVideo) {
      getAnnotations(currentVideo.id).then(setAnnotations);
    }
  }, [currentVideo]);

  return (
    <div className="App">
      <header className="App-header">
        <div className="flex justify-between items-center w-full max-w-7xl mx-auto px-4">
          <h1>SpokHand SLR Annotation Tool</h1>
          <div className="flex items-center gap-4">
            {currentUser ? (
              <>
                <span className="text-sm">Welcome, {currentUser.name}</span>
                <Button onClick={handleSignOut} variant="outline" size="sm">
                  Sign Out
                </Button>
              </>
            ) : (
              <Button onClick={() => handleSignIn('demo@spokhand.com', 'demo123')} size="sm">
                Sign In
              </Button>
            )}
          </div>
        </div>
      </header>

      <main className="p-4 max-w-7xl mx-auto">
        {!currentUser ? (
          <div className="text-center py-8">
            <p>Please sign in to start annotating</p>
          </div>
        ) : (
          <>
            {/* Session Management */}
            <div className="mb-6 p-4 border rounded-lg bg-gray-50">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold">Sessions</h2>
                <div className="flex gap-2">
                  <Button onClick={handleExportData} variant="outline" size="sm">
                    Export Data
                  </Button>
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleImportData}
                    className="hidden"
                    id="import-data"
                  />
                  <label htmlFor="import-data">
                    <Button variant="outline" size="sm" asChild>
                      <span>Import Data</span>
                    </Button>
                  </label>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {sessions.map(session => (
                  <div
                    key={session.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      currentSession?.id === session.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-100'
                    }`}
                    onClick={() => setCurrentSession(session)}
                  >
                    <h3 className="font-medium">{session.name}</h3>
                    <p className="text-sm text-gray-600">{session.description}</p>
                    <p className="text-xs text-gray-500 mt-2">
                      {session.videos.length} videos • {session.status}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Video Upload */}
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
                  {currentVideo && (
                    <Button onClick={handleAnalyzeVideo} disabled={isAnalyzing} variant="outline">
                      {isAnalyzing ? 'Analyzing...' : 'AI Analyze'}
                    </Button>
                  )}
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
          </>
        )}
      </main>
    </div>
  );
}

export default App;
