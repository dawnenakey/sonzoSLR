import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipBack, SkipForward, ChevronDown, RotateCcw } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { formatTime } from './timeUtils';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function VideoPlayer({ 
  videoUrl, 
  onTimeUpdate,
  videoRef: externalVideoRef,
  showMainControls = true,
  seekStepMode = "second",
  children,
  segmentStartTime,
  segmentDuration,
  loopSegment = false,
  autoPlaySegment = false,
  muted = false,
  containerClassName = "relative w-full h-full bg-black flex items-center justify-center",
  videoClassName = "max-w-full max-h-full object-contain"
}) {
  const internalVideoRef = useRef(null);
  const videoRef = externalVideoRef || internalVideoRef;

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(segmentStartTime !== undefined ? segmentStartTime : 0);
  const [duration, setDuration] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const segmentEndTimeoutRef = useRef(null);

  const FRAMES_PER_SECOND = 25;

  useEffect(() => {
    if (videoRef?.current) {
      videoRef.current.muted = muted;
      videoRef.current.playbackRate = playbackSpeed;
    }
  }, [videoRef, playbackSpeed, muted]);

  useEffect(() => {
    const player = videoRef.current;
    if (!player) return;

    const handleLoadedMeta = () => {
      setDuration(player.duration || 0);
      setIsLoaded(true);
      if (segmentStartTime !== undefined && player.currentTime !== segmentStartTime) {
        player.currentTime = segmentStartTime;
        setCurrentTime(segmentStartTime);
      }
      if (autoPlaySegment) {
        handlePlaySegment();
      }
    };
    
    player.addEventListener('loadedmetadata', handleLoadedMeta);
    player.addEventListener('play', () => setIsPlaying(true));
    player.addEventListener('pause', () => setIsPlaying(false));
    player.addEventListener('ended', () => {
      setIsPlaying(false);
      if (segmentStartTime !== undefined && loopSegment) { // If segment ended and loop is true
        handlePlaySegment(); // Replay segment
      }
    });
    player.addEventListener('timeupdate', handleVideoTimeUpdateInternal);
    player.addEventListener('error', handleError);

    // Set initial time if segmentStartTime is defined
    if (segmentStartTime !== undefined && player.readyState >= 1) { // HAVE_METADATA
         if (player.currentTime !== segmentStartTime) player.currentTime = segmentStartTime;
         setCurrentTime(segmentStartTime);
    }
    if (autoPlaySegment && player.readyState >= 3) { // HAVE_FUTURE_DATA
        handlePlaySegment();
    }


    return () => {
      player.removeEventListener('loadedmetadata', handleLoadedMeta);
      player.removeEventListener('play', () => setIsPlaying(true));
      player.removeEventListener('pause', () => setIsPlaying(false));
      player.removeEventListener('ended', () => setIsPlaying(false));
      player.removeEventListener('timeupdate', handleVideoTimeUpdateInternal);
      player.removeEventListener('error', handleError);
      if (segmentEndTimeoutRef.current) clearTimeout(segmentEndTimeoutRef.current);
    };
  }, [videoRef, videoUrl, segmentStartTime, autoPlaySegment]); // Re-run if segmentStartTime or autoPlaySegment changes


  const handleVideoTimeUpdateInternal = () => {
    if (videoRef?.current) {
      const time = videoRef.current.currentTime;
      setCurrentTime(time);
      if (onTimeUpdate) onTimeUpdate(time);

      // Enforce segment end if segment props are defined
      if (segmentStartTime !== undefined && segmentDuration !== undefined && !loopSegment) {
        const segmentEndTime = segmentStartTime + segmentDuration;
        if (time >= segmentEndTime - 0.05) { // Check slightly before to ensure pause
          if (!videoRef.current.paused) {
            videoRef.current.pause();
          }
          // Clear any pending timeout if manually paused or reached end
          if (segmentEndTimeoutRef.current) clearTimeout(segmentEndTimeoutRef.current);
        }
      }
    }
  };
  
  const handlePlaySegment = () => {
    const player = videoRef.current;
    if (!player || !isLoaded) return;

    const start = segmentStartTime !== undefined ? segmentStartTime : 0;
    player.currentTime = start; // Always seek to segment start (or video start)
    
    player.play().catch(e => console.warn("Video play failed:", e));

    if (segmentStartTime !== undefined && segmentDuration !== undefined) {
      if (segmentEndTimeoutRef.current) clearTimeout(segmentEndTimeoutRef.current);
      segmentEndTimeoutRef.current = setTimeout(() => {
        if (player && !player.paused) {
          if (loopSegment) {
            player.currentTime = segmentStartTime; // Loop back
            player.play().catch(console.warn);
          } else {
            player.pause();
          }
        }
      }, segmentDuration * 1000);
    }
  };

  const handlePlayPauseButtonClick = () => {
    const player = videoRef.current;
    if (!player || !isLoaded) return;

    if (player.paused || player.ended) {
      if (segmentStartTime !== undefined && segmentDuration !== undefined) {
        // If current time is outside segment or at the end, restart segment
        const segmentEndTime = segmentStartTime + segmentDuration;
        if (player.currentTime < segmentStartTime || player.currentTime >= segmentEndTime - 0.05) {
           handlePlaySegment();
        } else { // resume within segment
           player.play().catch(console.warn);
        }
      } else { // Normal playback
        player.play().catch(console.warn);
      }
    } else {
      player.pause();
      if (segmentEndTimeoutRef.current) clearTimeout(segmentEndTimeoutRef.current); // Clear timeout on manual pause
    }
  };

  const handleSkipButtonClick = (amount) => {
    if (!videoRef?.current || !duration) return;
    const player = videoRef.current;
    let targetTime = player.currentTime + amount;

    if (segmentStartTime !== undefined && segmentDuration !== undefined) {
        const segmentEndTime = segmentStartTime + segmentDuration;
        targetTime = Math.max(segmentStartTime, Math.min(segmentEndTime, targetTime));
    } else {
        targetTime = Math.max(0, Math.min(duration, targetTime));
    }
    player.currentTime = targetTime;
    setCurrentTime(targetTime);
    if (onTimeUpdate) onTimeUpdate(targetTime);
  };

  const handleSpeedChange = (speed) => { 
    setPlaybackSpeed(speed);
  };
  
  const handleError = (e) => {
    console.error('Video error:', e);
  };

  const formatTimeDisplay = (timeInSeconds) => {
    if (isNaN(timeInSeconds) || timeInSeconds === null || timeInSeconds === undefined) timeInSeconds = 0; 
    if (seekStepMode === 'frame') {
      const totalFrames = Math.max(0, Math.round(timeInSeconds * FRAMES_PER_SECOND));
      return `Frame ${String(totalFrames).padStart(3, '0')}`;
    } else {
      return formatTime(timeInSeconds);
    }
  };
  
  const currentDisplayTime = (segmentStartTime !== undefined && currentTime < segmentStartTime) 
                              ? segmentStartTime 
                              : currentTime;

  return (
    <>
      
      <div className={containerClassName}>
        <video
          ref={videoRef}
          className={videoClassName}
          src={videoUrl}
          onClick={showMainControls ? handlePlayPauseButtonClick : undefined} 
          playsInline
          controls={false} // Always false, custom controls are used
        >
          Your browser does not support the video tag.
        </video>
        {!isLoaded && videoUrl && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80 pointer-events-none">
            <div className="text-white flex flex-col items-center">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-t-transparent border-white mb-2"></div>
              <span className="text-sm">Loading...</span>
            </div>
          </div>
        )}
      </div>

      <div className="bg-[#121826] divide-y divide-gray-700/50">
        {children} 
        
        {showMainControls && isLoaded && (
          <div className={`px-4 pt-2 pb-1`}>
            <div className="flex items-center justify-between">
              <div className="text-sm font-mono text-white/90 min-w-[80px] text-left">
                {formatTimeDisplay(currentDisplayTime)}
              </div>
              
              <div className="flex items-center gap-1 sm:gap-2">
                {(segmentStartTime !== undefined && loopSegment) && (
                   <Button variant="ghost" size="icon" className="text-white/70 hover:text-white hover:bg-white/10 rounded-full h-8 w-8 sm:h-9 sm:w-9" title="Looping Segment">
                     <RotateCcw className="h-4 w-4 sm:h-5 sm:w-5" />
                   </Button>
                )}
                <Button variant="ghost" size="icon" onClick={() => handleSkipButtonClick( seekStepMode === 'frame' ? (-5/FRAMES_PER_SECOND) : -5)} className="text-white/90 hover:text-white hover:bg-white/10 rounded-full h-8 w-8 sm:h-9 sm:w-9">
                  <SkipBack className="h-4 w-4 sm:h-5 sm:w-5" />
                </Button>
                <Button variant="ghost" size="icon" onClick={handlePlayPauseButtonClick} className="text-white/90 hover:text-white hover:bg-white/10 rounded-full w-9 h-9 sm:w-10 sm:h-10">
                  {isPlaying ? <Pause className="h-5 w-5 sm:h-6 sm:w-6" /> : <Play className="h-5 w-5 sm:h-6 sm:w-6" />}
                </Button>
                <Button variant="ghost" size="icon" onClick={() => handleSkipButtonClick( seekStepMode === 'frame' ? (5/FRAMES_PER_SECOND) : 5)} className="text-white/90 hover:text-white hover:bg-white/10 rounded-full h-8 w-8 sm:h-9 sm:w-9">
                  <SkipForward className="h-4 w-4 sm:h-5 sm:w-5" />
                </Button>
              </div>
              
              <div className="flex items-center min-w-[80px] justify-end">
                <div className="text-sm font-mono text-white/90 pr-1 sm:pr-2">
                  {formatTimeDisplay(segmentStartTime !== undefined && segmentDuration !== undefined ? segmentStartTime + segmentDuration : duration)}
                </div>
                
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" className="text-white/90 hover:text-white hover:bg-white/10 px-1.5 py-1 sm:px-2 sm:py-1 h-7 text-xs">
                      {playbackSpeed}x <ChevronDown className="ml-1 h-3 w-3" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-20 bg-gray-800 border-gray-700 text-white">
                    {[2, 1.5, 1, 0.75, 0.5, 0.25].map((speed) => (
                      <DropdownMenuItem 
                        key={speed} 
                        className={`text-center cursor-pointer hover:bg-gray-700 ${playbackSpeed === speed ? 'bg-gray-600' : ''}`} 
                        onClick={() => handleSpeedChange(speed)}
                      >
                        {speed}x
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}