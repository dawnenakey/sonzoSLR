import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { createPageUrl } from '@/utils';
import { AnnotationEntity } from '@/api/entities';
import VideoPlayer from '../components/VideoPlayer';
import AnnotationTimeline from '../components/AnnotationTimeline';
import AnnotationControls from '../components/AnnotationControls';
import AnnotationList from '../components/AnnotationList';
import AnnotationDetailDialog from '../components/AnnotationDetailDialog';
import ExportJsonDialog from '../components/ExportJsonDialog';
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { TooltipProvider } from "@/components/ui/tooltip";
import { 
    ArrowLeft, Download, Trash2, Info, Video, Copy, HandMetal, Loader2, MoreVertical, Trash,
    Play, Pause as PauseIcon, CornerDownLeft, ArrowRight, XSquare, FileJson, ListTree, InfoIcon, RotateCcw, RotateCw
} from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { AnimatePresence, motion } from "framer-motion";
import Header from '../components/Header';
import { videoAPI } from '@/api/awsClient';

export default function Annotator() {
  const params = new URLSearchParams(window.location.search);
  const videoId = params.get('id');
  const navigate = useNavigate();
  
  const [videoData, setVideoData] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [segmentStartTime, setSegmentStartTime] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [isDetailDialogOpen, setIsDetailDialogOpen] = useState(false);
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
  const [segmentPlaybackActive, setSegmentPlaybackActive] = useState(false);
  const [playbackSegment, setPlaybackSegment] = useState(null);
  const [seekStepMode, setSeekStepMode] = useState("second");
  const [error, setError] = useState(null);
  const [isDeletingAll, setIsDeletingAll] = useState(false);
  
  const videoRef = useRef(null);
  const { toast } = useToast();

  const [annotationHistory, setAnnotationHistory] = useState([]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState(-1);
  const [lastAction, setLastAction] = useState(null);
  
  const [isCommandKeyPressed, setIsCommandKeyPressed] = useState(false);
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;

  useEffect(() => {
    if (!videoId) {
      setError("No video ID was provided in the URL. Please select a video from the library.");
      setIsLoading(false);
      return;
    }
    fetchData();

    return () => {
      setAnnotationHistory([]);
      setCurrentHistoryIndex(-1);
    };
  }, [videoId]);

  const fetchData = async () => {
    if (!videoId) {
        setError("No video ID was provided. Please select a video from the library.");
        setIsLoading(false);
        return;
    }
    try {
      setIsLoading(true);
      setError(null);
      console.log("fetchData: Fetching video and annotations for ID:", videoId);

      let video = null;
      try {
        video = await videoAPI.get(videoId);
      } catch (err) {
        console.error("Error fetching video:", err);
        if (err.message && err.message.includes("No item found with id")) {
             setError(`Could not load the video with ID "${videoId}". It may have been deleted or the ID is incorrect.`);
        } else {
            setError("Could not load the requested video. It may have been deleted or you may not have permission to view it.");
        }
      }

      if (!video && !error) {
        setError(`Could not load the video with ID "${videoId}". It might not exist.`);
      } else if (video) {
         setVideoData(video);
      }

      if (video) {
        let fetchedAnnotations = [];
        try {
            fetchedAnnotations = await AnnotationEntity.filter({ video_id: videoId }, 'start_time');
        } catch (err) {
            console.error("Error fetching annotations:", err);
            toast({
            variant: "warning",
            title: "Annotation data unavailable",
            description: "Could not load annotations. You can still create new ones."
            });
        }
        const initialAnnotations = Array.isArray(fetchedAnnotations) ? fetchedAnnotations : [];
        setAnnotations(initialAnnotations);
        console.log("fetchData: Initializing history with loaded annotations:", initialAnnotations);
        setAnnotationHistory([{ annotations: JSON.parse(JSON.stringify(initialAnnotations)), actionType: 'initial_load' }]);
        setCurrentHistoryIndex(0);
      }
    } catch (err) {
      console.error("Annotator initialization error:", err);
      if (!error) {
        setError("An unexpected error occurred while setting up the annotator.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleTimeUpdate = (time) => {
    setCurrentTime(time);
    if (isSegmenting && segmentStartTime !== null && time - segmentStartTime >= 0.1) {
      const displayElement = document.querySelector(".segment-duration-display");
      if (displayElement) {
          displayElement.classList.add("active");
      }
    }
  };

  const handleSegmentStart = () => {
    if (!videoData || !videoRef.current) return;
    setIsSegmenting(true);
    setSegmentStartTime(currentTime);
    const displayElement = document.querySelector(".segment-duration-display");
    if (displayElement) {
        displayElement.classList.remove("active");
    }
    
    if (videoRef.current && videoRef.current.paused) {
      videoRef.current.play().catch(e => console.warn("Play failed on segment start", e));
    }
    
    toast({
      title: "Recording segment...",
      description: "Press Enter to categorize and save. Press Esc to cancel.",
      duration: 5000,
    });
  };

  const handleSegmentEnd = (segmentType) => {
    if (!videoRef.current || !isSegmenting || segmentStartTime === null) return;
    
    const endTime = currentTime;
    const duration = endTime - segmentStartTime;
    
    if (duration < 0.1) {
      toast({
        variant: "destructive",
        title: "Segment too short",
        description: "Please make a longer segment (at least 0.1 seconds)."
      });
      setIsSegmenting(false);
      setSegmentStartTime(null);
      return; 
    }
    
    if (videoRef.current?.pause) videoRef.current.pause();

    const newAnnotationData = {
      video_id: videoData.id,
      segment_type: segmentType, // Now just passing 'SEGMENT' as the default type
      start_time: segmentStartTime,
      duration: duration,
      description: ""
    };
    
    saveAnnotation(newAnnotationData, 'create_segment'); 
    setIsSegmenting(false);
    setSegmentStartTime(null);
  };
  
  const handleCancelSegment = () => {
    setIsSegmenting(false);
    setSegmentStartTime(null);
    if (videoRef.current && !videoRef.current.paused) {
        videoRef.current.pause();
    }
    toast({
      title: "Segmentation cancelled",
      duration: 2000,
    });
  };

  const addToHistory = (currentAnnotationsState, actionType) => {
    console.log(`addToHistory called. Action: ${actionType}. currentHistoryIndex before: ${currentHistoryIndex}`);
    const newHistoryRecords = annotationHistory.slice(0, currentHistoryIndex + 1);
    
    const newStateEntry = {
      annotations: JSON.parse(JSON.stringify(currentAnnotationsState)),
      actionType
    };
    newHistoryRecords.push(newStateEntry);
    
    setAnnotationHistory(newHistoryRecords);
    setCurrentHistoryIndex(newHistoryRecords.length - 1);
    console.log(`addToHistory: New history length: ${newHistoryRecords.length}, new index: ${newHistoryRecords.length - 1}`);
  };

  const saveAnnotation = async (annotationData, actionTypeContext = 'create') => {
    try {
      setIsSaving(true);
      const savedAnnotation = await AnnotationEntity.create(annotationData);
      const newAnnotations = [...annotations, savedAnnotation].sort((a,b) => a.start_time - b.start_time);
      setAnnotations(newAnnotations);
      
      addToHistory(newAnnotations, actionTypeContext);
      
      let segmentTypeLabels = {
        'SEGMENT': 'Segment'
      };
      
      toast({
        title: `${segmentTypeLabels[annotationData.segment_type]} saved`,
        description: `Start: ${annotationData.start_time.toFixed(2)}s, Duration: ${annotationData.duration.toFixed(2)}s`,
      });
    } catch (err) {
      console.error("Error saving annotation:", err);
      toast({
        variant: "destructive",
        title: "Failed to save segment",
        description: "There was an error saving your annotation. Please try again."
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleUpdateAnnotation = async (updatedAnnotation, isFromResize = false) => {
    if(!updatedAnnotation || !updatedAnnotation.id) {
      toast({ variant: "destructive", title: "Error", description: "Cannot update annotation without ID."});
      return;
    }
    console.log("handleUpdateAnnotation: Updating annotation ID:", updatedAnnotation.id, "Data:", updatedAnnotation);
    try {
      setIsSaving(true);
      const newAnnotationsOptimistic = annotations.map(a => 
        a.id === updatedAnnotation.id ? updatedAnnotation : a
      ).sort((a,b) => a.start_time - b.start_time);
      setAnnotations(newAnnotationsOptimistic);

      await AnnotationEntity.update(updatedAnnotation.id, updatedAnnotation);
      
      addToHistory(newAnnotationsOptimistic, isFromResize ? 'resize_segment' : 'edit_segment_details');
      
      toast({
        title: "Annotation updated",
        description: "Your changes have been saved."
      });
      setSelectedSegment(updatedAnnotation);
      
      if (!isFromResize && videoRef.current?.pause) {
        videoRef.current.pause();
      }
    } catch (err) {
      console.error("Error updating annotation:", err);
      const originalAnnotations = annotations;
      setAnnotations(originalAnnotations);

      toast({
        variant: "destructive",
        title: "Failed to update segment",
        description: "There was an error updating your annotation. Please try again."
      });
       fetchData();
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteAnnotation = async (annotation) => {
    if(!annotation || !annotation.id) {
        toast({ variant: "destructive", title: "Error", description: "Cannot delete annotation without ID."});
        return;
    }
    try {
      await AnnotationEntity.delete(annotation.id);
      const newAnnotations = annotations.filter(a => a.id !== annotation.id);
      setAnnotations(newAnnotations);
      addToHistory(newAnnotations, 'delete_segment');
      
      if (selectedSegment && selectedSegment.id === annotation.id) {
        setSelectedSegment(null);
        setIsDetailDialogOpen(false);
      }
      
      toast({
        title: "Annotation deleted",
        description: "The segment has been removed."
      });
    } catch (err) {
    }
  };

  const handleDeleteAllAnnotations = async () => {
    if (!videoData || annotations.length === 0) {
      toast({
        variant: "info",
        title: "No Annotations",
        description: "There are no annotations to delete.",
      });
      setIsDeletingAll(false);
      return;
    }

    console.log(`handleDeleteAllAnnotations: Attempting to delete ${annotations.length} annotations for video ID: ${videoData.id}`);
    const originalAnnotations = JSON.parse(JSON.stringify(annotations)); // Keep a copy for potential immediate undo

    try {
      setIsSaving(true); // Use isSaving to indicate processing
      // Perform backend deletions
      // This assumes AnnotationEntity.delete can take an array of IDs or we loop
      // For simplicity, let's assume we loop, or a batch delete endpoint exists
      // This part needs to align with actual capabilities of AnnotationEntity
      for (const annotation of annotations) {
        await AnnotationEntity.delete(annotation.id);
      }

      const newAnnotations = []; // All annotations are gone
      setAnnotations(newAnnotations);
      addToHistory(newAnnotations, 'delete_all_segments'); // Add to history

      toast({
        title: "All Annotations Deleted",
        description: `Successfully removed ${originalAnnotations.length} annotations for "${videoData.title}".`,
      });
    } catch (err) {
      console.error("Error deleting all annotations:", err);
      // Revert local state if backend failed (or rely on undo)
      setAnnotations(originalAnnotations); // Simple revert, user can try again or undo if history was added.
      
      toast({
        variant: "destructive",
        title: "Deletion Failed",
        description: "Could not delete all annotations. Please try again.",
      });
    } finally {
      setIsSaving(false);
      setIsDeletingAll(false); // Close the confirmation dialog
    }
  };

  const handleUndo = async () => {
    console.log("handleUndo: Called. currentHistoryIndex:", currentHistoryIndex, "History length:", annotationHistory.length);
    if (currentHistoryIndex <= 0) {
      toast({ title: "Nothing to undo", variant: "info", duration: 1500 });
      console.log("handleUndo: Nothing to undo or at start of history.");
      return;
    }

    const previousStateIndex = currentHistoryIndex - 1;
    const previousState = annotationHistory[previousStateIndex];
    console.log("handleUndo: Attempting to revert to state at index", previousStateIndex, "Action type:", previousState.actionType);
    console.log("handleUndo: Current annotations before local revert:", JSON.parse(JSON.stringify(annotations)));
    console.log("handleUndo: Target annotations from history:", JSON.parse(JSON.stringify(previousState.annotations)));

    try {
      setAnnotations(JSON.parse(JSON.stringify(previousState.annotations)));
      setCurrentHistoryIndex(previousStateIndex);
      
      toast({
        title: "Undo successful",
        description: `Reverted: ${previousState.actionType || 'last action'}`,
        duration: 2000,
      });
      console.log("handleUndo: Local state updated. New currentHistoryIndex:", previousStateIndex);

      const currentAnnotationsBeforeUndoBackend = annotationHistory[currentHistoryIndex].annotations;
      const targetAnnotationsForBackend = previousState.annotations;

      for (const targetAnno of targetAnnotationsForBackend) {
        const existingInCurrent = currentAnnotationsBeforeUndoBackend.find(ca => ca.id === targetAnno.id);
        if (existingInCurrent) {
            if (JSON.stringify(existingInCurrent) !== JSON.stringify(targetAnno)) {
                 await AnnotationEntity.update(targetAnno.id, { ...targetAnno, video_id: videoId });
            }
        } else {
            await AnnotationEntity.create({ ...targetAnno, video_id: videoId });
        }
      }
      for (const currentAnno of currentAnnotationsBeforeUndoBackend) {
        if (!targetAnnotationsForBackend.find(ta => ta.id === currentAnno.id)) {
            await AnnotationEntity.delete(currentAnno.id);
        }
      }
      console.log("handleUndo: Backend synchronization attempted.");

    } catch (error) {
      console.error("Error during undo (local or backend sync):", error);
      toast({
        variant: "destructive",
        title: "Undo Partially Failed",
        description: "Local state reverted, but backend sync encountered an error. Data might be inconsistent.",
      });
    }
  };

  const handleRedo = async () => {
    console.log("handleRedo: Called. currentHistoryIndex:", currentHistoryIndex, "History length:", annotationHistory.length);
    if (currentHistoryIndex >= annotationHistory.length - 1 || currentHistoryIndex < 0) {
      toast({ title: "Nothing to redo", variant: "info", duration: 1500 });
      console.log("handleRedo: Nothing to redo or history issue.");
      return;
    }

    const nextStateIndex = currentHistoryIndex + 1;
    const nextState = annotationHistory[nextStateIndex];
    console.log("handleRedo: Attempting to advance to state at index", nextStateIndex, "Action type:", nextState.actionType);

    try {
      setAnnotations(JSON.parse(JSON.stringify(nextState.annotations)));
      setCurrentHistoryIndex(nextStateIndex);

      toast({
        title: "Redo successful",
        description: `Re-applied: ${nextState.actionType || 'next action'}`,
        duration: 2000,
      });
      console.log("handleRedo: Local state updated. New currentHistoryIndex:", nextStateIndex);
      
      const currentAnnotationsBeforeRedoBackend = annotationHistory[currentHistoryIndex -1 < 0 ? 0 : currentHistoryIndex].annotations;
      const targetAnnotationsForBackend = nextState.annotations;
       for (const targetAnno of targetAnnotationsForBackend) {
        const existingInCurrent = currentAnnotationsBeforeRedoBackend.find(ca => ca.id === targetAnno.id);
        if (existingInCurrent) {
            if (JSON.stringify(existingInCurrent) !== JSON.stringify(targetAnno)) {
                 await AnnotationEntity.update(targetAnno.id, { ...targetAnno, video_id: videoId });
            }
        } else {
            await AnnotationEntity.create({ ...targetAnno, video_id: videoId });
        }
      }
      for (const currentAnno of currentAnnotationsBeforeRedoBackend) {
        if (!targetAnnotationsForBackend.find(ta => ta.id === currentAnno.id)) {
            await AnnotationEntity.delete(currentAnno.id);
        }
      }
      console.log("handleRedo: Backend synchronization attempted.");

    } catch (error) {
      console.error("Error during redo (local or backend sync):", error);
      toast({
        variant: "destructive",
        title: "Redo Partially Failed",
        description: "Local state re-applied, but backend sync encountered an error. Data might be inconsistent.",
      });
    }
  };
  
  const handleEditAnnotation = (annotation) => {
    setSelectedSegment(annotation);
    setIsDetailDialogOpen(true);
  };

  const handleAnnotationClick = (annotation) => {
    setSelectedSegment(annotation);
    setIsDetailDialogOpen(true);
  };

  const handlePlayAnnotation = (annotation) => {
    if (!videoRef.current || !annotation) return;

    setSelectedSegment(annotation);
    setSegmentPlaybackActive(true);
    setPlaybackSegment(annotation);
    
    videoRef.current.currentTime = annotation.start_time;
    videoRef.current.play().catch(e => console.warn("Play failed on annotation click", e));
    
    if (window.segmentEndTimeout) {
      clearTimeout(window.segmentEndTimeout);
    }

    const segmentEndTime = annotation.start_time + annotation.duration;
    const timeToEnd = (segmentEndTime - videoRef.current.currentTime);
    
    if (timeToEnd > 0) {
        window.segmentEndTimeout = setTimeout(() => {
        if (videoRef.current && videoRef.current.currentTime >= segmentEndTime - 0.05 ) {
          videoRef.current.pause();
        }
      }, timeToEnd * 1000);
    } else {
        videoRef.current.pause();
        setSegmentPlaybackActive(false); 
        setPlaybackSegment(null);
    }
  };

  const handleTimelineSeek = (time) => {
    if (videoRef.current) {
      const safeDuration = videoRef.current?.duration || Infinity;
      const newTime = Math.max(0, Math.min(time || 0, safeDuration));
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime); 
      
      if (segmentPlaybackActive) {
        setSegmentPlaybackActive(false);
        setPlaybackSegment(null);
        if (window.segmentEndTimeout) {
          clearTimeout(window.segmentEndTimeout);
        }
      }
    }
  };

  const handleSeekStepModeChange = (mode) => {
    setSeekStepMode(mode);
  };

  const handlePauseVideo = () => {
    if (videoRef.current) {
      videoRef.current.pause();
    }
    if (segmentPlaybackActive) {
      setSegmentPlaybackActive(false);
      setPlaybackSegment(null);
      if (window.segmentEndTimeout) {
        clearTimeout(window.segmentEndTimeout);
      }
    }
  };

    const handlePlayPauseToggle = () => {
      if (!videoRef.current) return;
      
      if (videoRef.current.paused || videoRef.current.ended) {
        videoRef.current.play().catch(e => console.warn("Play failed", e));
      } else {
        videoRef.current.pause();
      }
    };
  
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Meta' || e.key === 'Control') {
        setIsCommandKeyPressed(true);
      }

      if (
        e.target.tagName === 'INPUT' || 
        e.target.tagName === 'TEXTAREA' || 
        e.target.isContentEditable ||
        isDetailDialogOpen || isExportDialogOpen
      ) {
        if (isCommandKeyPressed && e.key.toLowerCase() === 'z') {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            } else {
                e.preventDefault();
                if (e.shiftKey) {
                    handleRedo();
                } else {
                    handleUndo();
                }
            }
        }
        return;
      }
      
      if (isCommandKeyPressed && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          handleRedo();
        } else {
          handleUndo();
        }
        return;
      }
      
      if (isSegmenting) {
        if (e.code === 'Enter') { 
          e.preventDefault(); 
          handleSegmentEnd('SEGMENT'); 
        }
        else if (e.key === 'Escape') { 
          e.preventDefault(); 
          handleCancelSegment(); 
        }
      }
      else if (e.code === 'Enter' && !isSegmenting) {
        e.preventDefault(); 
        handleSegmentStart();
      }
      
      if (e.code === 'Space') {
        e.preventDefault();
        if (!isSegmenting) {
          handlePlayPauseToggle();
        }
      }
      
      const FRAMES_PER_SEC = videoData?.fps || 25;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (videoRef.current && videoData) {
          const step = seekStepMode === 'frame' ? 1/FRAMES_PER_SEC : 1;
          const newTime = Math.max(0, videoRef.current.currentTime - step);
          videoRef.current.currentTime = newTime;
          setCurrentTime(newTime);
        }
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (videoRef.current && videoData) {
          const step = seekStepMode === 'frame' ? 1/FRAMES_PER_SEC : 1;
          const newTime = Math.min(videoRef.current.duration || Infinity, videoRef.current.currentTime + step);
          videoRef.current.currentTime = newTime;
          setCurrentTime(newTime);
        }
      }
    };

    const handleKeyUp = (e) => {
      if (e.key === 'Meta' || e.key === 'Control') {
        setIsCommandKeyPressed(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [
    isSegmenting, segmentStartTime, currentTime, seekStepMode, videoData, 
    isDetailDialogOpen, isExportDialogOpen, 
    isCommandKeyPressed, currentHistoryIndex, annotationHistory,
    annotations
  ]);

  useEffect(() => {
    let intervalId;
    if (segmentPlaybackActive && playbackSegment && videoRef.current) {
      const videoElement = videoRef.current;
      const checkSegmentEnd = () => {
        if (!playbackSegment || !videoElement) {
             if (intervalId) clearInterval(intervalId);
             setSegmentPlaybackActive(false);
             setPlaybackSegment(null);
             return;
        }
        const segmentEndTime = playbackSegment.start_time + playbackSegment.duration;
        if (videoElement.currentTime >= segmentEndTime - 0.05 || videoElement.ended || (videoElement.paused && segmentPlaybackActive)) {
          if (videoElement.currentTime >= segmentEndTime - 0.05 && !videoElement.paused) {
            videoElement.pause();
          }
          clearInterval(intervalId);
          setSegmentPlaybackActive(false);
          setPlaybackSegment(null);
           if (window.segmentEndTimeout) {
            clearTimeout(window.segmentEndTimeout);
          }
        }
      };
      intervalId = setInterval(checkSegmentEnd, 50);
    }
    return () => {
        if (intervalId) clearInterval(intervalId);
        if (window.segmentEndTimeout) clearTimeout(window.segmentEndTimeout);
    };
  }, [segmentPlaybackActive, playbackSegment]);
  
  const currentSegmentDuration = isSegmenting && segmentStartTime !== null
    ? Math.max(0, (currentTime - segmentStartTime))
    : 0;
  
  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center flex flex-col items-center justify-center min-h-[calc(100vh-200px)]">
        <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
        <p className="text-lg text-muted-foreground">Loading video and annotations...</p>
      </div>
    );
  }


  if (error) {
    return (
      <div className="max-w-4xl mx-auto py-6 px-4">
        <Alert variant="destructive" className="mb-4">
          <Info className="h-5 w-5" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button onClick={() => navigate(createPageUrl('Home'))} variant="outline" className="my-2">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Return to Video Library
        </Button>
      </div>
    );
  }

  if (!videoData && !isLoading) {
    return (
      <div className="max-w-4xl mx-auto py-6 px-4">
        <Alert className="mb-4">
          <Info className="h-5 w-5" />
          <AlertTitle>Video Not Found</AlertTitle>
          <AlertDescription>The requested video could not be loaded. It might have been deleted or the ID is incorrect.</AlertDescription>
        </Alert>
        <Link to={createPageUrl('Home')}>
          <Button variant="outline" className="my-2">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Return to Video Library
          </Button>
        </Link>
      </div>
    );
  }
  
  if (!videoData) return null; 

  const ShortcutItem = ({ icon: Icon, text, shortcut, className }) => (
    <li className={`flex items-center justify-between text-sm ${className || ''}`}>
      <div className="flex items-center gap-2 text-gray-700">
        <Icon className="h-4 w-4 text-indigo-500" />
        <span>{text}</span>
      </div>
      <kbd className="font-mono bg-gray-200 text-gray-700 px-1.5 py-0.5 rounded text-xs border border-gray-300 shadow-sm">{shortcut}</kbd>
    </li>
  );

  return (
    <>
      <Header />
      <TooltipProvider>
        <div className="max-w-7xl mx-auto px-2 sm:px-4">
           <div className="flex items-center justify-between mb-4 border-b pb-2">
            <Link to={createPageUrl('Home')} className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 group">
              <ArrowLeft className="h-3.5 w-3.5 mr-1 transition-transform group-hover:-translate-x-1" />
              Back
            </Link>
            
            <h1 className="text-lg md:text-xl font-semibold truncate max-w-md mx-2 text-center">
              {videoData?.title || "Video Annotator"}
            </h1>
            
            <Button 
              size="sm"
              variant="outline" 
              className="flex items-center gap-1"
              onClick={() => setIsExportDialogOpen(true)}
            >
              <Download className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Export</span>
            </Button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">
            <div className="lg:col-span-8 space-y-4 sm:space-y-6">
              <div className="bg-white rounded-xl overflow-hidden shadow-lg">
                <VideoPlayer 
                  videoUrl={videoData?.url} 
                  videoRef={videoRef}
                  onTimeUpdate={handleTimeUpdate}
                  isPlayingSegment={segmentPlaybackActive}
                  seekStepMode={seekStepMode}
                >
                  {/* AnnotationControls injected into VideoPlayer */}
                  <AnnotationControls 
                    isSegmenting={isSegmenting}
                    onSegmentStart={handleSegmentStart}
                    currentSegmentDuration={currentSegmentDuration}
                    onSegmentEnd={handleSegmentEnd}
                    onCancelSegment={handleCancelSegment}
                  />
                </VideoPlayer>

                <AnnotationTimeline 
                  annotations={annotations}
                  videoDuration={videoRef.current?.duration ?? videoData?.duration ?? 0}
                  currentTime={currentTime}
                  onAnnotationClick={handlePlayAnnotation}
                  activeSegment={playbackSegment || selectedSegment}
                  onDeleteAnnotation={handleDeleteAnnotation}
                  onUpdateAnnotation={handleUpdateAnnotation}
                  onSeek={handleTimelineSeek}
                  seekStepMode={seekStepMode}
                  onSeekStepModeChange={handleSeekStepModeChange}
                  onPauseVideo={handlePauseVideo}
                />
              </div>
            </div>
            
            <div className="lg:col-span-4 space-y-4 sm:space-y-6">
              <div className="bg-white p-4 sm:p-5 rounded-xl shadow-lg">
                <h3 className="text-md font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <InfoIcon className="h-5 w-5 text-indigo-600" />
                  Controls & Shortcuts
                </h3>
                <div className="space-y-3">
                  <div>
                    <h4 className="text-xs font-medium text-gray-500 uppercase mb-1.5">Playback</h4>
                    <ul className="space-y-1">
                      <ShortcutItem icon={Play} text="Play / Pause" shortcut="Space" />
                      <ShortcutItem icon={ArrowRight} text="Seek Video" shortcut="← / →" />
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-xs font-medium text-gray-500 uppercase mb-1.5">Segmentation</h4>
                    <ul className="space-y-1">
                      <ShortcutItem icon={CornerDownLeft} text="New Segment" shortcut="Enter" />
                      <ShortcutItem icon={XSquare} text="Cancel" shortcut="Esc" />
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="text-xs font-medium text-gray-500 uppercase mb-1.5">Edit Actions</h4>
                    <ul className="space-y-1">
                      <ShortcutItem 
                        icon={RotateCcw} 
                        text="Undo" 
                        shortcut={isMac ? "⌘ Z" : "Ctrl + Z"} 
                      />
                      <ShortcutItem 
                        icon={RotateCw} 
                        text="Redo" 
                        shortcut={isMac ? "⌘ ⇧ Z" : "Ctrl + Shift + Z"} 
                      />
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-xs font-medium text-gray-500 uppercase mb-1.5">List Menu</h4>
                     <ul className="space-y-1">
                      <ShortcutItem icon={FileJson} text="Export JSON" shortcut="•••" />
                      <ShortcutItem icon={Trash2} text="Delete All" shortcut="•••" />
                    </ul>
                  </div>
                </div>
                 <p className="text-xs text-gray-500 mt-4 pt-2 border-t border-gray-200">
                  Annotations save automatically. Edit by clicking on timeline or list items.
                </p>
              </div>
            </div>

            <div className="lg:col-span-12 mt-0 sm:mt-0"> 
              <AnnotationList 
                annotations={annotations}
                onEditAnnotation={handleEditAnnotation}
                onDeleteAnnotation={handleDeleteAnnotation}
                onAnnotationClick={handleAnnotationClick}
                activeSegment={segmentPlaybackActive ? playbackSegment : selectedSegment}
                onTriggerExportJson={() => setIsExportDialogOpen(true)}
                onTriggerDeleteAll={() => setIsDeletingAll(true)}
              />
            </div>
          </div>

          <AnnotationDetailDialog 
            isOpen={isDetailDialogOpen}
            onClose={() => {setSelectedSegment(null); setIsDetailDialogOpen(false);}}
            annotation={selectedSegment}
            onSave={handleUpdateAnnotation}
            videoDuration={videoRef.current?.duration ?? videoData?.duration ?? 0}
            videoUrl={videoData?.url}
          />

          <ExportJsonDialog 
            isOpen={isExportDialogOpen}
            onClose={() => setIsExportDialogOpen(false)}
            annotations={annotations}
            videoTitle={videoData?.title || "video"}
          />
          <AlertDialog open={isDeletingAll} onOpenChange={setIsDeletingAll}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete All Annotations?</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to delete all {annotations.length} annotations for "{videoData?.title}"? 
                  This action can be undone locally using {isMac ? "⌘ Z" : "Ctrl + Z"} immediately after.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel onClick={() => setIsDeletingAll(false)}>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  className="bg-red-600 hover:bg-red-700"
                  onClick={handleDeleteAllAnnotations}
                >
                  Delete All
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>

        </div>
      </TooltipProvider>
    </>
  );
}
