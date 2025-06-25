import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { AlertCircle, ChevronLeft, ChevronRight, Play, Pause } from 'lucide-react';
import { Slider } from "@/components/ui/slider";
import { formatTime, parseTime } from './timeUtils';
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useToast } from "@/components/ui/use-toast";

export default function AnnotationDetailDialog({ 
  isOpen, 
  onClose, 
  annotation, 
  onSave,
  videoDuration,
  videoUrl,
  seekStepMode = 'second'
}) {
  const [currentAnnotation, setCurrentAnnotation] = useState(null);
  const [startTimeInput, setStartTimeInput] = useState('');
  const [endTimeInput, setEndTimeInput] = useState('');
  const [isValidTiming, setIsValidTiming] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef(null);
  const timeUpdateRef = useRef(null);

  const { toast } = useToast();
  const FRAMES_PER_SECOND = 25; 
  const isVideoAvailable = !!videoUrl;
  
  // Calculate percentage values for the slider
  const startPercent = currentAnnotation && videoDuration 
    ? (currentAnnotation.start_time / videoDuration) * 100 
    : 0;
    
  const endPercent = currentAnnotation && videoDuration 
    ? ((currentAnnotation.start_time + currentAnnotation.duration) / videoDuration) * 100 
    : 0;

  const formatTimeForInput = (timeInSeconds) => {
    if (seekStepMode === 'frame') {
      const totalFrames = Math.round(timeInSeconds * FRAMES_PER_SECOND);
      const hours = Math.floor(totalFrames / (3600 * FRAMES_PER_SECOND));
      const minutes = Math.floor((totalFrames % (3600 * FRAMES_PER_SECOND)) / (60 * FRAMES_PER_SECOND));
      const seconds = Math.floor((totalFrames % (60 * FRAMES_PER_SECOND)) / FRAMES_PER_SECOND);
      const frames = totalFrames % FRAMES_PER_SECOND;
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(frames).padStart(2, '0')}`;
    }
    return formatTime(timeInSeconds);
  };
  
  const parseTimeFromInput = (timeString) => {
    if (seekStepMode === 'frame' && timeString.includes('.')) {
      const parts = timeString.split('.');
      const hms = parts[0];
      const frames = parseInt(parts[1], 10);
      if (isNaN(frames)) return parseTime(hms);
      return parseTime(hms) + (frames / FRAMES_PER_SECOND);
    }
    return parseTime(timeString);
  };

  useEffect(() => {
    if (isOpen && annotation) {
      const initialAnnotationState = {
        ...annotation,
        segment_type: ["SIGN_UNIT", "BREAK", "FALSE_POSITIVE"].includes(annotation.segment_type) 
                      ? annotation.segment_type 
                      : "SIGN_UNIT"
      };
      setCurrentAnnotation(initialAnnotationState);
      setStartTimeInput(formatTimeForInput(annotation.start_time));
      setEndTimeInput(formatTimeForInput(annotation.start_time + annotation.duration));
      setIsValidTiming(true);
      setIsPlaying(false);
      
      // Reset video to start position when dialog opens
      if (videoRef.current) {
        videoRef.current.currentTime = annotation.start_time;
      }
    }

    // Cleanup on dialog close
    return () => {
      if (videoRef.current && !videoRef.current.paused) {
        videoRef.current.pause();
      }
      if (timeUpdateRef.current) {
        clearInterval(timeUpdateRef.current);
      }
    };
  }, [isOpen, annotation, seekStepMode]);

  // Setup video timeupdate handler when video is loaded
  useEffect(() => {
    if (videoRef.current && currentAnnotation) {
      const handleTimeUpdate = () => {
        const video = videoRef.current;
        if (!video) return;

        const startTime = currentAnnotation.start_time;
        const endTime = startTime + currentAnnotation.duration;

        // If playhead goes beyond segment end, pause and reset to start
        if (video.currentTime >= endTime) {
          video.pause();
          video.currentTime = startTime;
          setIsPlaying(false);
        }
      };

      // Add native timeupdate event listener
      videoRef.current.addEventListener('timeupdate', handleTimeUpdate);

      // Backup interval check for more reliable checking (browser inconsistencies)
      timeUpdateRef.current = setInterval(() => {
        if (videoRef.current && currentAnnotation) {
          const video = videoRef.current;
          const startTime = currentAnnotation.start_time;
          const endTime = startTime + currentAnnotation.duration;

          if (video.currentTime >= endTime) {
            video.pause();
            video.currentTime = startTime;
            setIsPlaying(false);
          }
        }
      }, 100);

      // Cleanup
      return () => {
        if (videoRef.current) {
          videoRef.current.removeEventListener('timeupdate', handleTimeUpdate);
        }
        if (timeUpdateRef.current) {
          clearInterval(timeUpdateRef.current);
        }
      };
    }
  }, [videoRef.current, currentAnnotation]);

  const validateAndUpdateTiming = (newStartTimeStr, newEndTimeStr) => {
    const start = parseTimeFromInput(newStartTimeStr);
    const end = parseTimeFromInput(newEndTimeStr);
    
    if (isNaN(start) || isNaN(end)) {
      setIsValidTiming(false); return false;
    }
    const minDuration = seekStepMode === 'frame' ? (1 / FRAMES_PER_SECOND) : 0.01;
    if (start >= end || start < 0 || end > (videoDuration + 0.01) || (end - start) < minDuration ) {
      setIsValidTiming(false); return false;
    }
    
    setIsValidTiming(true);
    setCurrentAnnotation(prev => ({ ...prev, start_time: start, duration: end - start }));
    
    // Update video position if timing changes
    if (videoRef.current) {
      videoRef.current.currentTime = start;
    }
    
    return true;
  };

  const handleSave = () => {
    // Validate one more time before saving
    if (!validateAndUpdateTiming(startTimeInput, endTimeInput)) {
      toast({
        title: "Invalid timing values",
        description: "Please correct the start and end times before saving.",
        variant: "destructive"
      });
      return;
    }
    
    onSave(currentAnnotation);
    onClose();
  };

  const handleTimeInputChange = (field, value) => {
    if (field === 'start') {
      setStartTimeInput(value);
    } else {
      setEndTimeInput(value);
    }
  };
  
  const handleTimeInputBlur = (field) => {
    validateAndUpdateTiming(
      field === 'start' ? startTimeInput : startTimeInput, 
      field === 'end' ? endTimeInput : endTimeInput
    );
  };
  
  const handleTimeInputFocus = () => {
    // Stop video playback when editing time values
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };
  
  const handleSegmentTypeChange = (value) => {
    setCurrentAnnotation(prev => ({
      ...prev,
      segment_type: value
    }));
  };

  const handleQualifiedChange = (checked) => {
    setCurrentAnnotation(prev => ({
      ...prev,
      qualified: checked
    }));
  };

  const handleDescriptionChange = (e) => {
    setCurrentAnnotation(prev => ({
      ...prev,
      description: e.target.value
    }));
  };

  const handleLabelChange = (e) => {
    setCurrentAnnotation(prev => ({
      ...prev,
      label: e.target.value
    }));
  };

  const handlePlaySegment = () => {
    if (!videoRef.current || !currentAnnotation) return;
    
    // Always set to start of segment
    videoRef.current.currentTime = currentAnnotation.start_time;
    
    // Play the video
    videoRef.current.play()
      .then(() => {
        setIsPlaying(true);
      })
      .catch(error => {
        console.error("Error playing video:", error);
        setIsPlaying(false);
      });
  };

  const handlePauseSegment = () => {
    if (!videoRef.current) return;
    
    videoRef.current.pause();
    setIsPlaying(false);
  };

  const handleTimeStep = (field, direction) => {
    const stepSize = seekStepMode === 'frame' ? (1 / FRAMES_PER_SECOND) : 0.1;
    
    if (field === 'start') {
      const newStartTime = Math.max(0, currentAnnotation.start_time + (direction * stepSize));
      if (newStartTime >= currentAnnotation.start_time + currentAnnotation.duration - stepSize) return;
      
      setStartTimeInput(formatTimeForInput(newStartTime));
      setCurrentAnnotation(prev => ({
        ...prev,
        start_time: newStartTime,
        duration: prev.start_time + prev.duration - newStartTime
      }));
      
      if (videoRef.current) videoRef.current.currentTime = newStartTime;
      
    } else {
      const newEndTime = currentAnnotation.start_time + currentAnnotation.duration + (direction * stepSize);
      if (newEndTime <= currentAnnotation.start_time + stepSize || newEndTime > videoDuration) return;
      
      setEndTimeInput(formatTimeForInput(newEndTime));
      setCurrentAnnotation(prev => ({
        ...prev,
        duration: newEndTime - prev.start_time
      }));
    }
  };
  
  const handleSliderChange = (value) => {
    if (!videoDuration || !currentAnnotation) return;
    
    const min = value[0] / 100 * videoDuration;
    const max = value[1] / 100 * videoDuration;
    
    if (min >= max - 0.01) return; // Prevent zero or negative duration
    
    setStartTimeInput(formatTimeForInput(min));
    setEndTimeInput(formatTimeForInput(max));
    setCurrentAnnotation(prev => ({
      ...prev,
      start_time: min,
      duration: max - min
    }));
    
    if (videoRef.current) videoRef.current.currentTime = min;
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Segment Details</DialogTitle>
          <DialogDescription>
            Review, qualify, and adjust segment information.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6 py-4">
          {/* Video section */}
          <div className="aspect-video bg-black rounded-md overflow-hidden relative">
            {isVideoAvailable ? (
              <video 
                ref={videoRef}
                className="w-full h-full object-contain"
                src={videoUrl}
                playsInline
                controls={false}
              >
                Your browser does not support the video tag.
              </video>
            ) : (
              <div className="flex flex-col items-center justify-center h-full bg-gray-800 text-white p-4 text-center">
                <AlertCircle className="h-10 w-10 mb-2 text-red-400" />
                <h3 className="text-lg font-medium mb-1">Video Unavailable</h3>
                <p className="text-sm text-gray-300">
                  The original video for this segment is not available.
                </p>
              </div>
            )}
          </div>
          
          {/* EXPLICIT PLAY/PAUSE BUTTONS */}
          {isVideoAvailable && (
            <div className="flex justify-center gap-4">
              <Button 
                variant={isPlaying ? "outline" : "default"}
                size="lg" 
                className={`flex items-center gap-2 ${isPlaying ? "" : "bg-indigo-600 hover:bg-indigo-700"}`}
                onClick={handlePlaySegment}
                disabled={isPlaying}
              >
                <Play className="h-5 w-5" /> Play Segment
              </Button>
              <Button 
                variant={isPlaying ? "default" : "outline"}
                size="lg"
                className={`flex items-center gap-2 ${isPlaying ? "bg-indigo-600 hover:bg-indigo-700" : ""}`}
                onClick={handlePauseSegment}
                disabled={!isPlaying}
              >
                <Pause className="h-5 w-5" /> Pause
              </Button>
            </div>
          )}
          
          {/* Segment range display and slider */}
          {isVideoAvailable && currentAnnotation && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-gray-500">
                <span>Segment Range</span>
                <span>{formatTimeForInput(currentAnnotation.start_time)} - {formatTimeForInput(currentAnnotation.start_time + currentAnnotation.duration)}</span>
              </div>
              <div className="relative pt-5 pb-1">
                <Slider
                  value={[startPercent, endPercent]}
                  min={0} max={100} step={0.1}
                  onValueChange={handleSliderChange}
                  className="!mt-0" disabled={!isVideoAvailable}
                />
                <div className="absolute inset-x-0 -top-1 flex justify-between text-xs text-gray-400">
                  <span>0:00</span>
                  <span>{formatTime(videoDuration)}</span>
                </div>
              </div>
            </div>
          )}
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="segment-start-time">Start Time</Label>
              <div className="flex items-center gap-1">
                <Button variant="outline" size="icon" className="h-9 w-9" onClick={() => handleTimeStep('start', -1)} disabled={!isVideoAvailable}><ChevronLeft className="h-4 w-4" /></Button>
                <Input
                  id="segment-start-time" value={startTimeInput}
                  onChange={(e) => handleTimeInputChange('start', e.target.value)}
                  onBlur={() => handleTimeInputBlur('start')}
                  onFocus={handleTimeInputFocus}
                  className={`font-mono h-9 ${!isValidTiming ? 'border-red-500' : ''}`}
                  placeholder={seekStepMode === 'frame' ? 'HH:MM:SS.FF' : 'HH:MM:SS.ms'}
                  disabled={!isVideoAvailable}
                />
                <Button variant="outline" size="icon" className="h-9 w-9" onClick={() => handleTimeStep('start', 1)} disabled={!isVideoAvailable}><ChevronRight className="h-4 w-4" /></Button>
              </div>
            </div>
            <div>
              <Label htmlFor="segment-end-time">End Time</Label>
              <div className="flex items-center gap-1">
                <Button variant="outline" size="icon" className="h-9 w-9" onClick={() => handleTimeStep('end', -1)} disabled={!isVideoAvailable}><ChevronLeft className="h-4 w-4" /></Button>
                <Input
                  id="segment-end-time" value={endTimeInput}
                  onChange={(e) => handleTimeInputChange('end', e.target.value)}
                  onBlur={() => handleTimeInputBlur('end')}
                  onFocus={handleTimeInputFocus}
                  className={`font-mono h-9 ${!isValidTiming ? 'border-red-500' : ''}`}
                  placeholder={seekStepMode === 'frame' ? 'HH:MM:SS.FF' : 'HH:MM:SS.ms'}
                  disabled={!isVideoAvailable}
                />
                <Button variant="outline" size="icon" className="h-9 w-9" onClick={() => handleTimeStep('end', 1)} disabled={!isVideoAvailable}><ChevronRight className="h-4 w-4" /></Button>
              </div>
            </div>
          </div>
          
          {!isValidTiming && (
            <div className="text-sm text-red-500 flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              Invalid timing values. Please check start and end times.
            </div>
          )}
          
          <div>
            <Label htmlFor="segment-type" className="mb-2 block">Segment Type</Label>
            <RadioGroup 
              id="segment-type"
              value={currentAnnotation?.segment_type || 'SIGN_UNIT'} 
              onValueChange={handleSegmentTypeChange}
              className="flex flex-wrap gap-6"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="SIGN_UNIT" id="sign-unit" />
                <Label htmlFor="sign-unit" className="cursor-pointer">Sign Unit</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="BREAK" id="break" />
                <Label htmlFor="break" className="cursor-pointer">Break</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="FALSE_POSITIVE" id="false-positive" />
                <Label htmlFor="false-positive" className="cursor-pointer">False Positive</Label>
              </div>
            </RadioGroup>
          </div>

          <div className="space-y-2">
            <Label htmlFor="segment-label">Label/Gloss</Label>
            <Input
              id="segment-label"
              value={currentAnnotation?.label || ''}
              onChange={handleLabelChange}
              placeholder="Optional: Enter a label or gloss (e.g., HELLO)"
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="segment-description">Description / Notes</Label>
            <Textarea
              id="segment-description"
              value={currentAnnotation?.description || ''}
              onChange={handleDescriptionChange}
              placeholder="Add notes or description for this segment"
              className="min-h-[100px]"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Checkbox 
              id="segment-qualified" 
              checked={currentAnnotation?.qualified || false}
              onCheckedChange={handleQualifiedChange}
            />
            <Label htmlFor="segment-qualified" className="text-sm">Mark as reviewed and validated</Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} className="mr-2">
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={!isValidTiming}>
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}