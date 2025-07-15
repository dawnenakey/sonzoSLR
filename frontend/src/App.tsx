import { useState, useEffect } from "react";
import "./App.css";
import { ImportVideoDialog } from "./components/ImportVideoDialog";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert";
import { Progress } from "./components/ui/progress";
import { 
  Camera, Video, Play, Pause, RotateCcw, Download, Upload, 
  CheckCircle, AlertCircle, Loader2, Eye, EyeOff, Settings,
  Zap, Brain, Search, FileVideo, Clock, BarChart3, Home, User, Settings as SettingsIcon,
  Briefcase
} from 'lucide-react';
import { useToast } from "./components/ui/use-toast";
import AnnotationControls from "./components/AnnotationControls";
import AnnotationList from "./components/AnnotationList";
import AnnotationTimeline from "./components/AnnotationTimeline";
import VideoPlayer from "./components/VideoPlayer";
import EnhancedVideoViewer from "./components/EnhancedVideoViewer";
import LiveCameraAnnotator from "./components/LiveCameraAnnotator";
import AdvancedSignSpotting from "./components/AdvancedSignSpotting";
import JobCoaching from "./pages/JobCoaching";
import { awsAPI } from './api/awsClient';

interface Annotation {
  id: string;
  startTime: number;
  endTime: number;
  label: string;
  confidence?: number;
  notes?: string;
}

interface VideoData {
  file: File;
  url: string;
  metadata: {
    name: string;
    size: number;
    type: string;
    source: string;
    timestamp: string;
  };
}

function App() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [currentSession, setCurrentSession] = useState<any | null>(null);
  const [currentVideo, setCurrentVideo] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [showAdvancedFeatures, setShowAdvancedFeatures] = useState(true);
  const [processedVideoData, setProcessedVideoData] = useState<VideoData | null>(null);
  const [currentView, setCurrentView] = useState('home'); // home, camera, analysis

  const { toast } = useToast();

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
    checkConnectionHealth();
  }, []);

  const checkConnectionHealth = async () => {
    try {
      // Check API connectivity
      const response = await fetch('/api/health', { 
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('degraded');
      }
    } catch (err) {
      setConnectionStatus('disconnected');
      console.warn('Connection check failed:', err);
    }
  };

  const loadSessions = async () => {
    try {
      // Note: awsAPI.sessions.list() might not exist, so we'll handle this gracefully
      // setSessions(await awsAPI.sessions.list());
      console.log('Sessions loading not implemented yet');
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
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
      toast({
        variant: "destructive",
        title: "No file selected",
        description: "Please select a video file to upload"
      });
      return;
    }
    setIsUploading(true);
    try {
      // Create a new session if none exists
      let session = currentSession;
      if (!session) {
        session = await awsAPI.sessions.create({ 
          name: selectedFile.name, 
          description: "Video uploaded from annotation tool" 
        });
        setCurrentSession(session);
        await loadSessions();
      }
      // Upload video to the session
      const video = await awsAPI.sessions.uploadVideo(session.id, selectedFile);
      console.log('Uploaded video object:', video);
      if (!video.id) {
        toast({
          variant: "destructive",
          title: "Upload Error",
          description: "Upload succeeded but video.id is missing!"
        });
      }
      setCurrentVideo(video);
      toast({
        title: "Upload Successful",
        description: "File uploaded successfully! You can now start annotating."
      });
    } catch (error) {
      console.error("Upload failed:", error);
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: `Upload failed: ${error}`
      });
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
      setAnnotations((prev: Annotation[]) => [...prev, annotation]);
      toast({
        title: "Annotation Created",
        description: "New annotation has been created successfully"
      });
    } catch (error) {
      console.error('Failed to create annotation:', error);
      toast({
        variant: "destructive",
        title: "Annotation Failed",
        description: "Failed to create annotation"
      });
    }
  };

  const handleUpdateAnnotation = async (annotationId: string, updates: any) => {
    if (!currentVideo) return;
    try {
      const updatedAnnotation = await awsAPI.annotations.update(annotationId, updates);
      setAnnotations((prev: Annotation[]) => prev.map((a: Annotation) => a.id === annotationId ? updatedAnnotation : a));
      toast({
        title: "Annotation Updated",
        description: "Annotation has been updated successfully"
      });
    } catch (error) {
      console.error('Failed to update annotation:', error);
      toast({
        variant: "destructive",
        title: "Update Failed",
        description: "Failed to update annotation"
      });
    }
  };

  const handleDeleteAnnotation = async (annotationId: string) => {
    if (!currentVideo) return;
    try {
      await awsAPI.annotations.delete(annotationId);
      setAnnotations((prev: Annotation[]) => prev.filter((a: Annotation) => a.id !== annotationId));
      toast({
        title: "Annotation Deleted",
        description: "Annotation has been deleted successfully"
      });
    } catch (error) {
      console.error('Failed to delete annotation:', error);
      toast({
        variant: "destructive",
        title: "Delete Failed",
        description: "Failed to delete annotation"
      });
    }
  };

  const handleVideoUploaded = async (file: File) => {
    try {
      setIsUploading(true);
      
      // Create video data object for enhanced viewer
      const videoData: VideoData = {
        file: file,
        url: URL.createObjectURL(file),
        metadata: {
          name: file.name,
          size: file.size,
          type: file.type,
          source: 'live-camera',
          timestamp: new Date().toISOString()
        }
      };
      
      setProcessedVideoData(videoData);
      
      toast({
        title: "Video Processed",
        description: "Video has been processed and is ready for analysis"
      });
      
    } catch (error) {
      console.error('Error processing video:', error);
      toast({
        variant: "destructive",
        title: "Processing Failed",
        description: "Could not process the uploaded video"
      });
    } finally {
      setIsUploading(false);
    }
  };

  // Load annotations when video changes
  useEffect(() => {
    if (currentVideo) {
      awsAPI.annotations.getByVideo(currentVideo.id).then(setAnnotations);
    }
  }, [currentVideo]);

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'disconnected': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'degraded': return 'Slow Connection';
      case 'disconnected': return 'Disconnected';
      default: return 'Unknown';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="w-full bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-blue-700">SpokHand SLR</h1>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`}></div>
                <span className="text-sm text-gray-600">{getConnectionStatusText()}</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAdvancedFeatures(!showAdvancedFeatures)}
                className="flex items-center space-x-2"
              >
                {showAdvancedFeatures ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                <span>{showAdvancedFeatures ? 'Hide' : 'Show'} Advanced</span>
              </Button>
              
              <div className="flex items-center space-x-2">
                <Button
                  variant={currentView === 'home' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setCurrentView('home')}
                >
                  <Home className="h-4 w-4 mr-2" />
                  Home
                </Button>
                <Button
                  variant={currentView === 'camera' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setCurrentView('camera')}
                >
                  <Camera className="h-4 w-4 mr-2" />
                  Camera
                </Button>
                <Button
                  variant={currentView === 'analysis' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setCurrentView('analysis')}
                >
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Analysis
                </Button>
                <Button
                  variant={currentView === 'coaching' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setCurrentView('coaching')}
                >
                  <Briefcase className="h-4 w-4 mr-2" />
                  Job Coaching
                </Button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentView === 'home' && (
          <div className="space-y-8">
            {/* Welcome Section */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-6 w-6 text-blue-600" />
                  <span>Advanced Sign Language Recognition</span>
                </CardTitle>
                <CardDescription>
                  Upload videos, record live camera feeds, and analyze sign language with advanced AI features
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center p-4 border rounded-lg">
                    <Upload className="h-8 w-8 mx-auto mb-2 text-blue-600" />
                    <h3 className="font-semibold">Upload Videos</h3>
                    <p className="text-sm text-gray-600">Import existing sign language videos for analysis</p>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <Camera className="h-8 w-8 mx-auto mb-2 text-green-600" />
                    <h3 className="font-semibold">Live Recording</h3>
                    <p className="text-sm text-gray-600">Record and analyze sign language in real-time</p>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <Brain className="h-8 w-8 mx-auto mb-2 text-purple-600" />
                    <h3 className="font-semibold">AI Analysis</h3>
                    <p className="text-sm text-gray-600">Advanced sign spotting and disambiguation</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Video Upload Section */}
            <Card>
              <CardHeader>
                <CardTitle>Upload Video</CardTitle>
                <CardDescription>
                  Import a video file for annotation and analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="video-container mx-auto bg-black rounded-lg overflow-hidden mb-4">
                  {videoUrl && <VideoPlayer videoUrl={videoUrl} onTimeUpdate={setCurrentTime} onDurationChange={setVideoDuration} />}
                </div>
                <div className="controls p-4 border rounded-md">
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
                        {isUploading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Uploading...
                          </>
                        ) : (
                          'Accept & Upload'
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Annotation Section */}
            {videoUrl && (
              <Card>
                <CardHeader>
                  <CardTitle>Annotation Tools</CardTitle>
                  <CardDescription>
                    Create and manage video annotations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
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
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {currentView === 'camera' && (
          <div className="space-y-8">
            <LiveCameraAnnotator 
              onVideoUploaded={handleVideoUploaded}
            />
            
            {processedVideoData && (
              <EnhancedVideoViewer
                videoData={processedVideoData}
                showAdvancedFeatures={showAdvancedFeatures}
                onVideoProcessed={(data) => {
                  console.log('Video processed:', data);
                }}
              />
            )}
          </div>
        )}

        {currentView === 'analysis' && (
          <div className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-6 w-6 text-purple-600" />
                  <span>Advanced Sign Spotting Analysis</span>
                </CardTitle>
                <CardDescription>
                  Configure and run advanced sign language recognition analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <AdvancedSignSpotting
                  videoRef={null}
                  analysisResults={null}
                  isAnalyzing={isAnalyzing}
                  onAnalyze={() => {
                    setIsAnalyzing(true);
                    // Simulate analysis
                    setTimeout(() => setIsAnalyzing(false), 3000);
                  }}
                />
              </CardContent>
            </Card>
          </div>
        )}

        {currentView === 'coaching' && (
          <JobCoaching />
        )}
      </main>
    </div>
  );
}

export default App;
