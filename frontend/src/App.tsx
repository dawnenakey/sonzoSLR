import React, { useState, useCallback, useEffect } from 'react';
import { Button } from './components/ui/button';
import { Upload, Download, LogIn, LogOut } from 'lucide-react';
import VideoPlayer from './components/VideoPlayer';
import AnnotationControls from './components/AnnotationControls';
import AnnotationList from './components/AnnotationList';
import AnnotationTimeline from './components/AnnotationTimeline';
import AnnotationDetailDialog from './components/AnnotationDetailDialog';
import ImportVideoDialog from './components/ImportVideoDialog';
import ExportJsonDialog from './components/ExportJsonDialog';
import { base44, signIn, signOut, getAuthStatus, initializeBase44 } from './api/base44Client';
import './App.css';

interface Annotation {
  id: string;
  startTime: number;
  endTime: number;
  label: string;
  notes?: string;
}

interface Video {
  id: string;
  name: string;
  url: string;
  duration: number;
}

function App() {
  const [videoSrc, setVideoSrc] = useState<string>('');
  const [videoName, setVideoName] = useState<string>('');
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [segmentStartTime, setSegmentStartTime] = useState(0);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string>();
  const [showAnnotationDialog, setShowAnnotationDialog] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [editingAnnotation, setEditingAnnotation] = useState<Annotation>();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);

  // Initialize Base44 on component mount
  useEffect(() => {
    const initBase44 = async () => {
      await initializeBase44();
      const authStatus = getAuthStatus();
      setIsAuthenticated(authStatus.isAuthenticated);
      setCurrentUser(authStatus.user);
    };
    initBase44();
  }, []);

  const handleSignIn = async () => {
    // For demo purposes, using mock credentials
    const result = await signIn('demo@example.com', 'password');
    if (result.success) {
      setIsAuthenticated(true);
      setCurrentUser(result.user);
    }
  };

  const handleSignOut = async () => {
    const result = await signOut();
    if (result.success) {
      setIsAuthenticated(false);
      setCurrentUser(null);
      setVideoSrc('');
      setAnnotations([]);
      setCurrentVideo(null);
    }
  };

  const handleImportVideo = useCallback(async (file: File) => {
    try {
      // Upload file to Base44
      const uploadResult = await base44.integrations.Core.UploadFile(file);
      
      // Create video entity in Base44
      const videoData = {
        name: file.name,
        url: uploadResult.url,
        duration: 0, // Will be updated when video loads
        userId: currentUser?.id || 'user-123'
      };
      
      const video = await base44.entities.Video.create(videoData);
      setCurrentVideo(video);
      
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setVideoName(file.name);
      setAnnotations([]);
      
      console.log('Video imported successfully:', video);
    } catch (error) {
      console.error('Failed to import video:', error);
      // Fallback to local file
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setVideoName(file.name);
      setAnnotations([]);
    }
  }, [currentUser]);

  const handleSegmentStart = useCallback(() => {
    setIsSegmenting(true);
    setSegmentStartTime(currentTime);
  }, [currentTime]);

  const handleSegmentEnd = useCallback(() => {
    setIsSegmenting(false);
    setShowAnnotationDialog(true);
  }, []);

  const handleCancelSegment = useCallback(() => {
    setIsSegmenting(false);
    setSegmentStartTime(0);
  }, []);

  const handleAnnotationSave = useCallback(async (data: { label: string; notes: string }) => {
    try {
      if (editingAnnotation) {
        // Update existing annotation in Base44
        const updatedAnnotation = await base44.entities.Annotation.update(
          editingAnnotation.id,
          data
        );
        
        setAnnotations(annotations.map(a =>
          a.id === editingAnnotation.id
            ? { ...a, ...data }
            : a
        ));
        setEditingAnnotation(undefined);
      } else {
        // Create new annotation in Base44
        const annotationData = {
          startTime: segmentStartTime,
          endTime: currentTime,
          label: data.label,
          notes: data.notes,
          videoId: currentVideo?.id || 'video-123',
          userId: currentUser?.id || 'user-123'
        };
        
        const newAnnotation = await base44.entities.Annotation.create(annotationData);
        
        const localAnnotation: Annotation = {
          id: newAnnotation.id,
          startTime: segmentStartTime,
          endTime: currentTime,
          label: data.label,
          notes: data.notes
        };
        
        setAnnotations([...annotations, localAnnotation]);
      }
      setShowAnnotationDialog(false);
    } catch (error) {
      console.error('Failed to save annotation:', error);
      // Fallback to local storage
      if (editingAnnotation) {
        setAnnotations(annotations.map(a =>
          a.id === editingAnnotation.id
            ? { ...a, ...data }
            : a
        ));
        setEditingAnnotation(undefined);
      } else {
        const newAnnotation: Annotation = {
          id: Math.random().toString(36).substr(2, 9),
          startTime: segmentStartTime,
          endTime: currentTime,
          label: data.label,
          notes: data.notes
        };
        setAnnotations([...annotations, newAnnotation]);
      }
      setShowAnnotationDialog(false);
    }
  }, [annotations, currentTime, segmentStartTime, editingAnnotation, currentVideo, currentUser]);

  const handleAnnotationEdit = useCallback((annotation: Annotation) => {
    setEditingAnnotation(annotation);
    setShowAnnotationDialog(true);
  }, []);

  const handleAnnotationDelete = useCallback(async (id: string) => {
    try {
      // Delete from Base44
      await base44.entities.Annotation.delete(id);
      setAnnotations(annotations.filter(a => a.id !== id));
      if (selectedAnnotationId === id) {
        setSelectedAnnotationId(undefined);
      }
    } catch (error) {
      console.error('Failed to delete annotation:', error);
      // Fallback to local deletion
      setAnnotations(annotations.filter(a => a.id !== id));
      if (selectedAnnotationId === id) {
        setSelectedAnnotationId(undefined);
      }
    }
  }, [annotations, selectedAnnotationId]);

  const handleAnnotationSelect = useCallback((annotation: Annotation) => {
    setSelectedAnnotationId(annotation.id);
  }, []);

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center space-y-4">
          <h1 className="text-2xl font-bold">Sign Language Annotation Tool</h1>
          <p className="text-muted-foreground">Please sign in to continue</p>
          <Button onClick={handleSignIn}>
            <LogIn className="mr-2 h-4 w-4" />
            Sign In (Demo)
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container mx-auto py-8 px-4">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">Sign Language Annotation Tool</h1>
            {currentUser && (
              <p className="text-sm text-muted-foreground">
                Welcome, {currentUser.name} ({currentUser.email})
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => setShowImportDialog(true)}
            >
              <Upload className="mr-2 h-4 w-4" />
              Import Video
            </Button>
            <Button
              variant="outline"
              onClick={() => setShowExportDialog(true)}
              disabled={annotations.length === 0}
            >
              <Download className="mr-2 h-4 w-4" />
              Export Annotations
            </Button>
            <Button
              variant="outline"
              onClick={handleSignOut}
            >
              <LogOut className="mr-2 h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>

        {videoSrc ? (
          <div className="grid grid-cols-3 gap-8">
            <div className="col-span-2 space-y-4">
              <VideoPlayer
                src={videoSrc}
                onTimeUpdate={setCurrentTime}
                onDurationChange={setVideoDuration}
              />
              <AnnotationTimeline
                annotations={annotations}
                duration={videoDuration}
                currentTime={currentTime}
                onSeek={setCurrentTime}
                selectedAnnotationId={selectedAnnotationId}
              />
              <AnnotationControls
                isSegmenting={isSegmenting}
                onSegmentStart={handleSegmentStart}
                currentSegmentDuration={
                  isSegmenting ? currentTime - segmentStartTime : 0
                }
                onSegmentEnd={handleSegmentEnd}
                onCancelSegment={handleCancelSegment}
              />
            </div>
            <div>
              <h2 className="text-lg font-semibold mb-4">Annotations</h2>
              <AnnotationList
                annotations={annotations}
                onEdit={handleAnnotationEdit}
                onDelete={handleAnnotationDelete}
                onSelect={handleAnnotationSelect}
                selectedAnnotationId={selectedAnnotationId}
              />
            </div>
          </div>
        ) : (
          <div className="text-center py-16">
            <p className="text-muted-foreground mb-4">
              Import a video to start annotating
            </p>
            <Button onClick={() => setShowImportDialog(true)}>
              <Upload className="mr-2 h-4 w-4" />
              Import Video
            </Button>
          </div>
        )}
      </div>

      <ImportVideoDialog
        isOpen={showImportDialog}
        onClose={() => setShowImportDialog(false)}
        onImport={handleImportVideo}
      />

      <AnnotationDetailDialog
        isOpen={showAnnotationDialog}
        onClose={() => {
          setShowAnnotationDialog(false);
          setEditingAnnotation(undefined);
        }}
        onSave={handleAnnotationSave}
        initialData={editingAnnotation ? {
          label: editingAnnotation.label,
          notes: editingAnnotation.notes || ''
        } : undefined}
      />

      <ExportJsonDialog
        isOpen={showExportDialog}
        onClose={() => setShowExportDialog(false)}
        annotations={annotations}
        videoMetadata={
          videoSrc
            ? {
                name: videoName,
                duration: videoDuration
              }
            : undefined
        }
      />
    </div>
  );
}

export default App;
