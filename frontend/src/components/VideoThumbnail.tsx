import React, { useRef, useEffect } from 'react';

interface VideoThumbnailProps {
  src: string;
  time?: number;
  width?: number;
  height?: number;
  className?: string;
}

export default function VideoThumbnail({
  src,
  time = 0,
  width = 160,
  height = 90,
  className = ''
}: VideoThumbnailProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const updateThumbnail = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Set canvas dimensions to match video aspect ratio
      const aspectRatio = video.videoWidth / video.videoHeight;
      canvas.width = width;
      canvas.height = width / aspectRatio;

      // Draw the video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    };

    video.addEventListener('loadeddata', () => {
      video.currentTime = time;
    });

    video.addEventListener('seeked', updateThumbnail);

    return () => {
      video.removeEventListener('seeked', updateThumbnail);
    };
  }, [src, time, width]);

  return (
    <>
      <canvas
        ref={canvasRef}
        className={`bg-black rounded-lg ${className}`}
        style={{ width, height }}
      />
      <video
        ref={videoRef}
        src={src}
        style={{ display: 'none' }}
        preload="auto"
      />
    </>
  );
} 