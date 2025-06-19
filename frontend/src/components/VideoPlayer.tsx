import React, { useRef, useEffect, useState } from 'react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

interface VideoPlayerProps {
  src: string;
  onTimeUpdate?: (time: number) => void;
  onDurationChange?: (duration: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
}

export default function VideoPlayer({
  src,
  onTimeUpdate,
  onDurationChange,
  onPlay,
  onPause
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      onTimeUpdate?.(video.currentTime);
    };

    const handleDurationChange = () => {
      setDuration(video.duration);
      onDurationChange?.(video.duration);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('durationchange', handleDurationChange);
    video.addEventListener('play', () => setIsPlaying(true));
    video.addEventListener('pause', () => setIsPlaying(false));

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('durationchange', handleDurationChange);
      video.removeEventListener('play', () => setIsPlaying(true));
      video.removeEventListener('pause', () => setIsPlaying(false));
    };
  }, [onTimeUpdate, onDurationChange]);

  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      video.play();
      onPlay?.();
    } else {
      video.pause();
      onPause?.();
    }
  };

  const seek = (time: number) => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = time;
  };

  const skipBackward = () => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = Math.max(0, video.currentTime - 5);
  };

  const skipForward = () => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = Math.min(video.duration, video.currentTime + 5);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-2">
      <video
        ref={videoRef}
        src={src}
        className="w-full aspect-video bg-black rounded-lg"
      />
      <div className="space-y-2">
        <div className="flex items-center justify-between px-1">
          <span className="text-sm text-muted-foreground">
            {formatTime(currentTime)}
          </span>
          <span className="text-sm text-muted-foreground">
            {formatTime(duration)}
          </span>
        </div>
        <Slider
          value={[currentTime]}
          min={0}
          max={duration}
          step={0.1}
          onValueChange={([value]) => seek(value)}
          className="w-full"
        />
        <div className="flex items-center justify-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={skipBackward}
          >
            <SkipBack className="h-4 w-4" />
            <span className="sr-only">Skip backward</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={togglePlayPause}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            <span className="sr-only">
              {isPlaying ? 'Pause' : 'Play'}
            </span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={skipForward}
          >
            <SkipForward className="h-4 w-4" />
            <span className="sr-only">Skip forward</span>
          </Button>
        </div>
      </div>
    </div>
  );
} 