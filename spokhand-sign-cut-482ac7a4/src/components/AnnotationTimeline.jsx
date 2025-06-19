
import React, { useRef, useEffect, useMemo, useState } from 'react';
import { motion } from "framer-motion";
import { formatTime } from './timeUtils'; // Standard time formatting
import { 
    ChevronLeft, 
    ChevronRight, 
    ZoomIn, 
    ZoomOut, 
    Maximize2, 
    Clock, 
    FrameIcon, 
    PauseIcon as LucidePauseIcon,
    HandMetal, 
    CornerUpRightIcon, 
    Info 
} from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

const FRAMES_PER_SECOND = 25; // Assuming a default, should ideally come from video metadata if available
const SECOND_INCREMENT_FOR_RESIZE = 0.1; 
const MIN_TIMELINE_VIEWPORT_WIDTH = 300; 

export default function AnnotationTimeline({ 
  annotations = [], 
  videoDuration = 0, 
  currentTime = 0, 
  onAnnotationClick = () => {}, 
  activeSegment = null, 
  onDeleteAnnotation = () => {}, 
  onUpdateAnnotation = () => {}, 
  onSeek = () => {}, 
  timelineStyle = "",
  seekStepMode = "second", 
  onSeekStepModeChange = () => {}, 
  onPauseVideo = () => {}
}) {
  const timelineRef = useRef(null);
  const timelineContentRef = useRef(null);
  
  const [timelineViewportWidth, setTimelineViewportWidth] = useState(MIN_TIMELINE_VIEWPORT_WIDTH);
  const [isDraggingSegmentEdge, setIsDraggingSegmentEdge] = useState(false);
  const [dragAnnotation, setDragAnnotation] = useState(null); // Live visual state during drag
  const [initialDragState, setInitialDragState] = useState(null); // Snapshot at mousedown
  const [dragType, setDragType] = useState(null); 
  const [zoomLevel, setZoomLevel] = useState(1); 
  const [isScrubbingTimeline, setIsScrubbingTimeline] = useState(false);
  const [selectedSegmentForKeyboardDelete, setSelectedSegmentForKeyboardDelete] = useState(null);
  const [activeEdgeDragTime, setActiveEdgeDragTime] = useState(null); 

  // Add a debug mode state for troubleshooting
  const [debugMode, setDebugMode] = useState(false);
  
  const [dragStartClientX, setDragStartClientX] = useState(null);
  const [dragStartTime, setDragStartTime] = useState(null);
  
  const effectiveVideoDuration = videoDuration > 0 
    ? videoDuration 
    : (annotations.length > 0 
        ? Math.max(...annotations.map(a => (a?.start_time || 0) + (a?.duration || 0)), 0.1)
        : 1);

  useEffect(() => {
    const timelineElement = timelineRef.current;
    if (timelineElement) {
      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          setTimelineViewportWidth(entry.contentRect.width);
        }
      });
      resizeObserver.observe(timelineElement);
      setTimelineViewportWidth(timelineElement.clientWidth); 

      return () => resizeObserver.unobserve(timelineElement);
    }
  }, []);

  useEffect(() => {
    // Toggle debug mode with Alt+D
    const handleKeyDown = (e) => {
      if (e.key === 'd' && e.altKey) {
        e.preventDefault();
        setDebugMode(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const timelineContentActualWidth = Math.max(timelineViewportWidth, MIN_TIMELINE_VIEWPORT_WIDTH) * zoomLevel;

  const segmentTypeColors = { 
    'SEGMENT': { bg: 'bg-blue-500', text: 'text-blue-50', border: 'border-blue-700', hoverBg: 'hover:bg-blue-600' },
    'SIGN_UNIT': { bg: 'bg-blue-500', text: 'text-blue-50', border: 'border-blue-700', hoverBg: 'hover:bg-blue-600' }, 
    'BREAK': { bg: 'bg-amber-500', text: 'text-amber-50', border: 'border-amber-700', hoverBg: 'hover:bg-amber-600' }, 
    'TRANSITION': { bg: 'bg-purple-500', text: 'text-purple-50', border: 'border-purple-700', hoverBg: 'hover:bg-purple-600' },
    'FALSE_POSITIVE': { bg: 'bg-gray-400', text: 'text-gray-50', border: 'border-gray-500', hoverBg: 'hover:bg-gray-500' }
  };

  const segmentTypeDefaultLabels = {
    'SEGMENT': 'Segment',
    'SIGN_UNIT': 'Sign',
    'BREAK': 'Break',
    'TRANSITION': 'Transition',
    'FALSE_POSITIVE': 'False Positive'
  };

  const sortedAnnotations = useMemo(() => 
    [...(annotations || [])].sort((a, b) => (a?.start_time || 0) - (b?.start_time || 0)), 
    [annotations]
  );

  const annotationsWithRows = useMemo(() => {
    if (!annotations || annotations.length === 0) return [];
    
    const rows = [];
    return sortedAnnotations.map(annotation => {
      if (!annotation) return { row: 0 }; 
      
      let rowIndex = 0;
      while (true) {
        if (!rows[rowIndex]) rows[rowIndex] = [];
        const overlapping = rows[rowIndex].some(existing => 
          ((annotation?.start_time || 0) < ((existing?.start_time || 0) + (existing?.duration || 0))) && 
          ((annotation?.start_time || 0) + (annotation?.duration || 0) > (existing?.start_time || 0))
        );
        if (!overlapping) {
          rows[rowIndex].push(annotation);
          break;
        }
        rowIndex++;
      }
      return { ...annotation, row: rowIndex };
    });
  }, [sortedAnnotations]);

  const calculateVisibleTimeRange = () => {
    if (!timelineRef.current || effectiveVideoDuration === 0) return { start: 0, end: 0 };
    
    const scrollLeft = timelineRef.current.scrollLeft;
    const containerClientWidth = timelineRef.current.clientWidth; 
    const timePerPixel = effectiveVideoDuration / timelineContentActualWidth;
    const visibleStartTime = scrollLeft * timePerPixel;
    const visibleEndTime = (scrollLeft + containerClientWidth) * timePerPixel;
    
    return { start: visibleStartTime, end: visibleEndTime };
  };

  useEffect(() => {
    if (timelineRef.current && videoDuration > 0 && !isScrubbingTimeline && !isDraggingSegmentEdge) {
      const timelineWrapper = timelineRef.current;
      const timeMarker = timelineWrapper.querySelector('.current-time-marker');

      if (timeMarker) {
        const markerPositionInContent = (currentTime / effectiveVideoDuration) * timelineContentActualWidth;
        const wrapperScrollLeft = timelineWrapper.scrollLeft;
        const wrapperClientWidth = timelineWrapper.clientWidth;

        if (markerPositionInContent < wrapperScrollLeft || markerPositionInContent > wrapperScrollLeft + wrapperClientWidth -1 ) {
          timelineWrapper.scrollTo({
            left: Math.max(0, markerPositionInContent - wrapperClientWidth / 2),
            behavior: 'smooth'
          });
        }
      }
    }
  }, [currentTime, videoDuration, annotationsWithRows?.length, zoomLevel, isScrubbingTimeline, isDraggingSegmentEdge, timelineContentActualWidth, effectiveVideoDuration]);

  // REIMPLEMENTED DRAG LOGIC - focused on direct cursor-to-handle relationship
  useEffect(() => {
    if (!isDraggingSegmentEdge || !dragAnnotation || !initialDragState) return;
    
    // Calculate time from mouse position, independent of zoom level
    const clientXToTime = (clientX) => {
      if (!timelineContentRef.current) return null;
      
      // If we have a starting reference point, use it for relative movement
      if (dragStartClientX !== null && dragStartTime !== null) {
        // Calculate pixel movement from drag start
        const pixelDelta = clientX - dragStartClientX;
        
        // Convert pixel delta to time delta
        const timePerPixel = effectiveVideoDuration / timelineContentActualWidth;
        const timeDelta = pixelDelta * timePerPixel;
        
        // Apply delta to the original starting time
        return dragStartTime + timeDelta;
      }
      
      // Fallback to absolute position calculation if no reference (shouldn't happen)
      const rect = timelineContentRef.current.getBoundingClientRect();
      const scrollLeft = timelineRef.current?.scrollLeft || 0;
      const mouseX = clientX - rect.left + scrollLeft;
      return (mouseX / timelineContentActualWidth) * effectiveVideoDuration;
    };
    
    const snapTimeToIncrement = (time) => {
      if (seekStepMode === 'frame') {
        const frameNumber = Math.round(time * FRAMES_PER_SECOND);
        return frameNumber / FRAMES_PER_SECOND;
      } else {
        return Math.round(time / SECOND_INCREMENT_FOR_RESIZE) * SECOND_INCREMENT_FOR_RESIZE;
      }
    };
    
    const handleMouseMove = (e) => {
      e.preventDefault();
      if (!timelineContentRef.current || effectiveVideoDuration <= 0) return;
      
      // Get current time based on mouse position, using relative movement
      const currentMouseTime = clientXToTime(e.clientX);
      if (currentMouseTime === null) return;
      
      // Define constraints
      const minTime = 0;
      const maxTime = effectiveVideoDuration;
      const minDuration = seekStepMode === 'frame' ? (1/FRAMES_PER_SECOND) : SECOND_INCREMENT_FOR_RESIZE;
      
      // Create updated segment state
      const updatedSegment = { ...dragAnnotation };
      
      // Original segment boundaries from initial state
      const origStart = initialDragState.start_time;
      const origEnd = origStart + initialDragState.duration;
      
      // Snap time and apply constraints
      let snappedMouseTime = snapTimeToIncrement(currentMouseTime);
      snappedMouseTime = Math.max(minTime, Math.min(maxTime, snappedMouseTime));
      
      // Update segment based on which handle is being dragged
      if (dragType === 'start') {
        // Dragging start handle (left edge)
        
        // New start time can't be after original end - minimum duration
        const maxStartTime = origEnd - minDuration;
        updatedSegment.start_time = Math.min(snappedMouseTime, maxStartTime);
        
        // Duration is adjusted based on new start time
        updatedSegment.duration = origEnd - updatedSegment.start_time;
        
        // Update seek position to the start time
        if (onSeek) onSeek(updatedSegment.start_time);
        setActiveEdgeDragTime(updatedSegment.start_time);
      } 
      else if (dragType === 'end') {
        // Dragging end handle (right edge)
        
        // New end time can't be before original start + minimum duration
        const minEndTime = origStart + minDuration;
        const newEndTime = Math.max(snappedMouseTime, minEndTime);
        
        // Keep start fixed, adjust duration
        updatedSegment.start_time = origStart;
        updatedSegment.duration = Math.min(newEndTime - origStart, maxTime - origStart);
        
        // Update seek position to the end time
        if (onSeek) onSeek(updatedSegment.start_time + updatedSegment.duration);
        setActiveEdgeDragTime(updatedSegment.start_time + updatedSegment.duration);
      }
      
      // Update visual state
      setDragAnnotation(updatedSegment);
      
      // Pause video during drag
      if (onPauseVideo) onPauseVideo();
      
      // Debug logging if enabled
      if (debugMode) {
        console.log({
          dragType,
          clientX: e.clientX,
          dragStartClientX,
          pixelDelta: e.clientX - dragStartClientX,
          mouseTime: currentMouseTime,
          snappedTime: snappedMouseTime,
          origStart,
          origEnd,
          newStart: updatedSegment.start_time,
          newEnd: updatedSegment.start_time + updatedSegment.duration,
          duration: updatedSegment.duration
        });
      }
    };
    
    const handleMouseUp = (e) => {
      e.preventDefault();
      
      if (!dragAnnotation) {
        // Clean up if no drag annotation
        setIsDraggingSegmentEdge(false);
        setDragAnnotation(null);
        setInitialDragState(null);
        setDragType(null);
        setActiveEdgeDragTime(null);
        setDragStartClientX(null);
        setDragStartTime(null);
        return;
      }
      
      // Create final validated annotation
      const finalAnnotation = { ...dragAnnotation };
      
      // Define min/max constraints
      const minDuration = seekStepMode === 'frame' ? (1/FRAMES_PER_SECOND) : SECOND_INCREMENT_FOR_RESIZE;
      
      // Perform final validation
      // 1. Ensure start time is within bounds
      finalAnnotation.start_time = Math.max(0, finalAnnotation.start_time);
      
      // 2. Ensure duration meets minimum requirement
      finalAnnotation.duration = Math.max(minDuration, finalAnnotation.duration);
      
      // 3. Ensure end time doesn't exceed video duration
      if (finalAnnotation.start_time + finalAnnotation.duration > effectiveVideoDuration) {
        finalAnnotation.duration = effectiveVideoDuration - finalAnnotation.start_time;
      }
      
      // 4. Handle the case where start time + min duration would exceed video duration
      if (finalAnnotation.duration < minDuration) {
        finalAnnotation.start_time = Math.max(0, effectiveVideoDuration - minDuration);
        finalAnnotation.duration = Math.min(minDuration, effectiveVideoDuration - finalAnnotation.start_time);
      }
      
      // Save the final annotation
      if (onUpdateAnnotation) {
        handleAnnotationUpdate(finalAnnotation, true);
      }
      
      // Set final seek position based on which handle was dragged
      const finalSeekPosition = dragType === 'start' 
        ? finalAnnotation.start_time 
        : finalAnnotation.start_time + finalAnnotation.duration;
      
      if (onSeek) {
        requestAnimationFrame(() => {
          onSeek(Math.max(0, Math.min(finalSeekPosition, effectiveVideoDuration)));
        });
      }
      
      // Reset all drag state
      setIsDraggingSegmentEdge(false);
      setDragAnnotation(null);
      setInitialDragState(null);
      setDragType(null);
      setActiveEdgeDragTime(null);
      setDragStartClientX(null);
      setDragStartTime(null);
    };
    
    // Add event listeners
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    // Cleanup
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [
    isDraggingSegmentEdge,
    dragAnnotation,
    initialDragState,
    dragType,
    effectiveVideoDuration,
    timelineContentActualWidth,
    seekStepMode,
    onUpdateAnnotation,
    onSeek,
    onPauseVideo,
    debugMode,
    dragStartClientX, // Include references to new drag tracking state
    dragStartTime
  ]);

  useEffect(() => {
    if (!isScrubbingTimeline) return;
    
    const handleMouseMove = (e) => {
      if (!timelineContentRef.current || !onSeek || effectiveVideoDuration === 0) return;
      const rect = timelineContentRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left + (timelineRef.current?.scrollLeft || 0), timelineContentActualWidth));
      const clickRatio = x / timelineContentActualWidth;
      const seekTime = clickRatio * effectiveVideoDuration;
      onSeek(seekTime);
    };
    
    const handleMouseUp = () => setIsScrubbingTimeline(false);
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isScrubbingTimeline, effectiveVideoDuration, onSeek, timelineContentActualWidth]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Delete' && selectedSegmentForKeyboardDelete && onDeleteAnnotation) {
        e.preventDefault();
        onDeleteAnnotation(selectedSegmentForKeyboardDelete);
        setSelectedSegmentForKeyboardDelete(null);
      } else if (e.key === 'Escape') {
        setSelectedSegmentForKeyboardDelete(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedSegmentForKeyboardDelete, onDeleteAnnotation]);

  const handleTimelineMouseDown = (e) => {
    if (e.target.closest('.annotation-segment-block, .resize-handle, .zoom-control, .reset-zoom, .mode-toggle')) return;

    if (timelineContentRef.current && onSeek && effectiveVideoDuration > 0) {
      if(onPauseVideo) onPauseVideo(); 
      const rect = timelineContentRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + (timelineRef.current?.scrollLeft || 0);
      const clickRatio = Math.max(0, Math.min(x / timelineContentActualWidth, 1));
      const seekTime = clickRatio * effectiveVideoDuration;
      
      if (seekTime <= effectiveVideoDuration) { 
        onSeek(seekTime);
        setIsScrubbingTimeline(true);
      }
    }
  };

  const handleZoomIn = () => setZoomLevel(prev => Math.min(prev * 1.5, 10));
  const handleZoomOut = () => setZoomLevel(prev => Math.max(prev / 1.5, 0.5));
  const handleResetZoom = () => setZoomLevel(1);
  const handleZoomSliderChange = (values) => setZoomLevel(values[0]);
  const handleToggleSeekStepMode = () => onSeekStepModeChange(seekStepMode === 'frame' ? 'second' : 'frame');

  const getTimeInterval = () => {
    if (effectiveVideoDuration === 0) return 1; 
    const visibleTimeRange = calculateVisibleTimeRange();
    const visibleDuration = Math.max(0.1, visibleTimeRange.end - visibleTimeRange.start); 
    
    if (seekStepMode === 'frame') {
      const visibleFrames = visibleDuration * FRAMES_PER_SECOND;
      let frameStep;
      if (visibleFrames <= 30) frameStep = 1;       // Show every frame if very zoomed in (e.g., less than ~1 sec visible)
      else if (visibleFrames <= 75) frameStep = 5;  // Every 5 frames
      else if (visibleFrames <= 150) frameStep = 10; // Every 10 frames
      else if (visibleFrames <= 375) frameStep = 25; // Every 25 frames (equivalent to 1 second marks at 25fps)
      else if (visibleFrames <= 750) frameStep = 50; // Every 50 frames
      else if (visibleFrames <= 1500) frameStep = 100; // Every 100 frames
      else frameStep = 250;                         // Fallback for larger zoom outs
      return frameStep / FRAMES_PER_SECOND; // Return the time equivalent of the frame step
    } else { 
      // For second mode, use existing logic
      if (visibleDuration <= 10) return 1;
      if (visibleDuration <= 30) return 2;
      if (visibleDuration <= 60) return 5;
      if (visibleDuration <= 300) return 10;
      if (visibleDuration <= 600) return 30;
      return 60;
    }
  };

  const formatTimeForMarker = (seconds) => {
    if (seekStepMode === 'frame') {
      const frameNumber = Math.round(seconds * FRAMES_PER_SECOND);
      return `f${frameNumber}`; // Simple frame count
    }
    // Default to normal time format for second mode
    return formatTime(seconds); // Uses the existing formatTime util for HH:MM:SS.ms
  };
  
  const renderVideoEndMarker = () => {
    if (!(videoDuration > 0)) return null; 
    return (
      <div 
        className="absolute top-0 h-full border-r-2 border-red-400 z-[15]" 
        style={{ left: `100%` }} 
      >
        <div className="absolute -top-5 right-0 translate-x-1/2 bg-red-500 text-white text-[10px] px-1.5 py-0.5 rounded-md shadow-md">
          End
        </div>
      </div>
    );
  };
  
  if (!(videoDuration > 0) && annotations.length === 0 && effectiveVideoDuration <=1 ) { 
    return (
      <div className={`px-3 sm:px-4 pt-1 pb-3 sm:pb-4 bg-[#121826]`}>
        <div className="h-24 sm:h-32 flex items-center justify-center bg-gray-800/50 rounded-lg border border-dashed border-gray-700">
          <p className="text-sm text-gray-400 px-4 text-center">Timeline empty. Video loading or no annotations.</p>
        </div>
      </div>
    );
  }
  
  const maxRow = annotationsWithRows?.length > 0 ? Math.max(...annotationsWithRows.map(a => a.row)) : 0;
  const timelineHeight = Math.max(80, (maxRow + 1) * 50); 
  const timeInterval = getTimeInterval();

  const handleAnnotationUpdate = (annotation, isFromResize = false) => {
    onUpdateAnnotation(annotation, isFromResize);
    
    // Add visual feedback for the updated annotation
    const element = document.querySelector(`[data-annotation-id="${annotation.id}"]`);
    if (element) {
      element.classList.add('annotation-updated');
      setTimeout(() => {
        element.classList.remove('annotation-updated');
      }, 300);
    }
  };

  // Update renderAnnotations to include special handling for end segments
  const renderAnnotations = () => {
    return annotationsWithRows?.map((annotation, index) => {
      if (effectiveVideoDuration === 0 || !annotation) return null; 
      
      // Check if this segment ends at the timeline's END
      const endsAtTimelineEnd = Math.abs(
        (annotation.start_time + annotation.duration) - effectiveVideoDuration
      ) < 0.001; // Small tolerance for floating point comparison
      
      const leftPosition = (annotation.start_time / effectiveVideoDuration) * 100;
      const minWidthPx = 5; 
      
      const annotationEndTime = Math.min(annotation.start_time + annotation.duration, effectiveVideoDuration);
      const displayDuration = Math.max(0, annotationEndTime - annotation.start_time);

      const calculatedWidthPercentage = (displayDuration / effectiveVideoDuration) * 100;
      const minWidthPercentage = (minWidthPx / timelineContentActualWidth) * 100;
      const widthPercentage = Math.max(calculatedWidthPercentage, minWidthPercentage);
      
      const colors = segmentTypeColors[annotation.segment_type] || segmentTypeColors['BREAK'];
      const isActive = (activeSegment && activeSegment.id === annotation.id) || 
        (selectedSegmentForKeyboardDelete && selectedSegmentForKeyboardDelete.id === annotation.id);
      const verticalPosition = annotation.row * 38; 
      
      const isBeingDragged = dragAnnotation && dragAnnotation.id === annotation.id;

      let currentDisplayLeft = leftPosition;
      let currentDisplayWidth = widthPercentage;

      if (isBeingDragged) {
        // Live updates during dragging
        currentDisplayLeft = (dragAnnotation.start_time / effectiveVideoDuration) * 100;
        const draggedDisplayDuration = Math.min(dragAnnotation.start_time + dragAnnotation.duration, effectiveVideoDuration) - dragAnnotation.start_time;
        currentDisplayWidth = (Math.max(0, draggedDisplayDuration) / effectiveVideoDuration) * 100;
      }

      const segmentLabel = annotation.description 
        ? (annotation.description.length > 20 ? annotation.description.substring(0, 17) + "..." : annotation.description) 
        : (segmentTypeDefaultLabels[annotation.segment_type] || "Segment");

      if (annotation.start_time >= effectiveVideoDuration && effectiveVideoDuration > 0) return null;

      return (
        <motion.div
          key={`anno-${annotation.id || index}`}
          initial={{ opacity: 0, y: 10 }} 
          animate={{ opacity: 1, y: 0 }} 
          transition={{ delay: index * 0.03 }}
          className={`annotation-segment-block absolute h-9 ${colors.bg} ${colors.hoverBg} rounded 
            ${isActive ? 'ring-2 ring-emerald-400/70 shadow-lg' : `border ${colors.border}`} 
            ${endsAtTimelineEnd && zoomLevel > 1 ? 'end-segment' : ''} 
            group cursor-pointer transition-all duration-200 shadow-sm flex items-center justify-start px-2 overflow-hidden z-10`}
          style={{ 
            left: `${currentDisplayLeft}%`, 
            width: `${currentDisplayWidth}%`, 
            top: `${verticalPosition}px`, 
            minWidth: `${minWidthPx}px`, 
            transform: isActive ? 'scale(1.02)' : 'scale(1)' 
          }}
          onClick={(e) => {
            e.stopPropagation();
            onAnnotationClick(annotation);
            setSelectedSegmentForKeyboardDelete(annotation);
          }}
          data-annotation-id={annotation.id}
          data-ends-at-end={endsAtTimelineEnd}
          data-zoom-level={zoomLevel.toFixed(1)}
        >
          {React.createElement( 
            { SEGMENT: Info, SIGN_UNIT: HandMetal, BREAK: LucidePauseIcon, TRANSITION: CornerUpRightIcon, FALSE_POSITIVE: Info }[annotation.segment_type] || Info, 
            { className: `h-3.5 w-3.5 ${colors.text} mr-1.5 flex-shrink-0` } 
          )}
          <span className={`text-[9px] sm:text-[10px] font-medium whitespace-nowrap ${colors.text} truncate`}>
            {segmentLabel}
          </span>
          
          <div 
            className={`resize-handle absolute left-0 top-0 bottom-0 w-3 cursor-w-resize 
              ${colors.text} opacity-30 hover:opacity-100 group-hover:opacity-70 group-hover:bg-black/30 hover:bg-black/40 
              transition-all duration-150 ${isActive ? 'opacity-70 bg-black/20' : ''}`}
            onMouseDown={(e) => { 
              e.stopPropagation();
              
              // Calculate the time position of this handle for relative movement
              const handleTime = annotation.start_time;
              
              // Store initial mouse position
              setDragStartClientX(e.clientX);
              setDragStartTime(handleTime);
              
              if (onPauseVideo) onPauseVideo();
              setIsDraggingSegmentEdge(true);
              setDragAnnotation({...annotation});
              setInitialDragState({...annotation});
              setDragType('start');
              setSelectedSegmentForKeyboardDelete(annotation);
              setActiveEdgeDragTime(handleTime);
              
              // Debug info
              if (debugMode) {
                console.log("START DRAG:", {
                  type: 'start',
                  id: annotation.id,
                  clientX: e.clientX,
                  handleTime,
                  zoom: zoomLevel,
                  endsAtEnd: Math.abs((annotation.start_time + annotation.duration) - effectiveVideoDuration) < 0.001
                });
              }
            }}
          >
            <div className="absolute top-0 bottom-0 left-0 right-0 flex items-center justify-center">
              <ChevronLeft className="h-4 w-4 drop-shadow-md" />
            </div>
            
            {((isDraggingSegmentEdge && dragType === 'start' && dragAnnotation?.id === annotation.id) || 
              (isActive && !isDraggingSegmentEdge)) && (
              <div className="absolute -top-7 -left-2 bg-gray-900 text-white text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap z-30 pointer-events-none">
                Start: {formatTime(isDraggingSegmentEdge && dragAnnotation ? dragAnnotation.start_time : annotation.start_time)}
              </div>
            )}
          </div>
          
          <div 
            className={`resize-handle absolute right-0 top-0 bottom-0 w-3 cursor-e-resize 
              ${colors.text} opacity-30 hover:opacity-100 group-hover:opacity-70 group-hover:bg-black/30 hover:bg-black/40
              transition-all duration-150 ${isActive ? 'opacity-70 bg-black/20' : ''}`}
            onMouseDown={(e) => { 
              e.stopPropagation();
              
              // Calculate the time position of this handle for relative movement
              const handleTime = annotation.start_time + annotation.duration;
              
              // Store initial mouse position
              setDragStartClientX(e.clientX);
              setDragStartTime(handleTime);
              
              if (onPauseVideo) onPauseVideo();
              setIsDraggingSegmentEdge(true);
              setDragAnnotation({...annotation});
              setInitialDragState({...annotation});
              setDragType('end');
              setSelectedSegmentForKeyboardDelete(annotation);
              setActiveEdgeDragTime(handleTime);
              
              // Debug info
              if (debugMode) {
                console.log("END DRAG:", {
                  type: 'end',
                  id: annotation.id, 
                  clientX: e.clientX,
                  handleTime,
                  zoom: zoomLevel,
                  endsAtEnd: Math.abs((annotation.start_time + annotation.duration) - effectiveVideoDuration) < 0.001
                });
              }
            }}
          >
            <div className="absolute top-0 bottom-0 left-0 right-0 flex items-center justify-center">
              <ChevronRight className="h-4 w-4 drop-shadow-md" />
            </div>
            
            {((isDraggingSegmentEdge && dragType === 'end' && dragAnnotation?.id === annotation.id) || 
              (isActive && !isDraggingSegmentEdge)) && (
              <div className="absolute -top-7 -right-2 bg-gray-900 text-white text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap z-30 pointer-events-none">
                End: {formatTime(isDraggingSegmentEdge && dragAnnotation ? (dragAnnotation.start_time + dragAnnotation.duration) : (annotation.start_time + annotation.duration))}
              </div>
            )}
          </div>
          
          {isDraggingSegmentEdge && dragAnnotation?.id === annotation.id && (
            <div className="absolute -bottom-7 left-1/2 -translate-x-1/2 bg-gray-900 text-white text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap z-30 pointer-events-none">
              Duration: {dragAnnotation.duration.toFixed(seekStepMode === 'frame' ? 3 : 2)}s 
              {seekStepMode === 'frame' && ` (${Math.round(dragAnnotation.duration * FRAMES_PER_SECOND)}f)`}
            </div>
          )}
        </motion.div>
      );
    });
  };

  return (
    <div className="relative">
      <style>{`
        @keyframes highlight-update {
          0% { transform: scale(1); }
          50% { transform: scale(1.02); }
          100% { transform: scale(1); }
        }
        .annotation-updated {
          animation: highlight-update 0.3s ease-in-out;
        }
        .resize-handle:hover::after {
          content: '';
          position: absolute;
          top: 0;
          bottom: 0;
          left: 0;
          right: 0;
          box-shadow: 0 0 0 1px white;
        }
        .end-segment {
          /* Add custom styling for segments at the end if needed */
        }
      `}</style>
      <div className={`px-3 sm:px-4 pt-1 pb-3 sm:pb-4 bg-[#121826] rounded-b-xl`}>
        <div className="mb-2 flex items-center justify-between">
          {/* Mode toggle on the left side - Redesigned */}
          <div className="flex items-center rounded-md border border-gray-700 overflow-hidden">
            <Button
              variant="ghost"
              size="sm"
              className={`h-7 px-2 text-xs rounded-none items-center gap-1 ${seekStepMode === 'frame' ? 'bg-sky-600 text-white hover:bg-sky-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'}`}
              onClick={() => onSeekStepModeChange('frame')}
              title="Frame Mode"
            >
              <FrameIcon className="h-3.5 w-3.5" /> <span className="hidden sm:inline">Frame</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={`h-7 px-2 text-xs rounded-none border-l border-gray-700 items-center gap-1 ${seekStepMode === 'second' ? 'bg-sky-600 text-white hover:bg-sky-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'}`}
              onClick={() => onSeekStepModeChange('second')}
              title="Second Mode"
            >
              <Clock className="h-3.5 w-3.5" /> <span className="hidden sm:inline">Second</span>
            </Button>
          </div>
          
          {/* Zoom controls on the right */}
          <div className="flex items-center gap-2 bg-gray-800/50 rounded-full px-2 py-1">
            <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-300 hover:text-white hover:bg-gray-700/50 rounded-full zoom-control" onClick={handleZoomOut}><ZoomOut className="h-3.5 w-3.5" /><span className="sr-only">Zoom Out</span></Button>
            <Slider value={[zoomLevel]} min={0.5} max={10} step={0.1} className="w-20 zoom-control" onValueChange={handleZoomSliderChange}/>
            <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-300 hover:text-white hover:bg-gray-700/50 rounded-full zoom-control" onClick={handleZoomIn}><ZoomIn className="h-3.5 w-3.5" /><span className="sr-only">Zoom In</span></Button>
            <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-300 hover:text-white hover:bg-gray-700/50 rounded-full reset-zoom" onClick={handleResetZoom}><Maximize2 className="h-3.5 w-3.5" /><span className="sr-only">Reset Zoom</span></Button>
            <span className="text-[10px] text-gray-400 font-mono whitespace-nowrap">{zoomLevel.toFixed(1)}x</span>
          </div>
        </div>
        
        <div 
          ref={timelineRef} 
          className="relative overflow-x-auto bg-gray-800 rounded-lg border border-gray-700 p-2 sm:p-4 cursor-default active:cursor-grabbing"
          style={{ height: `${timelineHeight + 40}px` }} // +40 for top/bottom padding and marker text space
          onMouseDown={handleTimelineMouseDown}
        >
          <div 
            ref={timelineContentRef} 
            className="relative timeline-content" 
            style={{ width: `${timelineContentActualWidth}px`, height: `${timelineHeight}px` }}
          >
             {effectiveVideoDuration > 0 && Array.from({ length: Math.ceil(effectiveVideoDuration / Math.max(timeInterval, 1/FRAMES_PER_SECOND)) + 1 }).map((_, i) => {
              const markerTime = i * Math.max(timeInterval, 1/FRAMES_PER_SECOND); // timeInterval is now correctly calculated for frames too
              if (markerTime > effectiveVideoDuration + (1/FRAMES_PER_SECOND / 2)) return null; 
              
              return (
                <div 
                  key={`marker-${i}`} 
                  className="absolute top-0 h-full border-l border-gray-600/70 flex flex-col items-center z-0" 
                  style={{ left: `${(markerTime / effectiveVideoDuration) * 100}%` }}
                >
                  <span className="text-[9px] sm:text-[10px] text-gray-400 absolute -top-4 bg-[#121826] px-0.5 rounded">
                    {formatTimeForMarker(markerTime)}
                  </span>
                </div>
              );
            })}
            
            {renderVideoEndMarker()}
            
            <div 
              className="absolute top-0 h-full border-l-2 border-red-500 z-20 current-time-marker pointer-events-none"
              style={{ left: `${effectiveVideoDuration > 0 ? (currentTime / effectiveVideoDuration) * 100 : 0}%` }}
            >
              <div className="absolute -top-5 -translate-x-1/2 bg-red-600 text-white text-[10px] sm:text-xs px-1.5 py-0.5 rounded-md shadow-md font-mono">
                {/* Simplified red playback bar label for frame mode */}
                {seekStepMode === 'frame' 
                  ? `f${Math.round(currentTime * FRAMES_PER_SECOND)}`
                  : formatTime(currentTime) // Existing format for second mode
                }
              </div>
            </div>

            {renderAnnotations()}
          </div>
        </div>
      </div>
    </div>
  );
}
