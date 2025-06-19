import React, { useRef, useEffect } from 'react';

interface Annotation {
  id: string;
  startTime: number;
  endTime: number;
  label: string;
}

interface AnnotationTimelineProps {
  annotations: Annotation[];
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
  selectedAnnotationId?: string;
}

export default function AnnotationTimeline({
  annotations,
  duration,
  currentTime,
  onSeek,
  selectedAnnotationId
}: AnnotationTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw annotations
    annotations.forEach((annotation) => {
      const startX = (annotation.startTime / duration) * canvas.width;
      const endX = (annotation.endTime / duration) * canvas.width;
      const width = endX - startX;

      // Draw segment
      ctx.fillStyle = selectedAnnotationId === annotation.id ? '#22c55e' : '#4b5563';
      ctx.fillRect(startX, 0, width, canvas.height);

      // Draw border
      ctx.strokeStyle = selectedAnnotationId === annotation.id ? '#16a34a' : '#374151';
      ctx.lineWidth = 1;
      ctx.strokeRect(startX, 0, width, canvas.height);
    });

    // Draw playhead
    const playheadX = (currentTime / duration) * canvas.width;
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(playheadX - 1, 0, 2, canvas.height);
  }, [annotations, duration, currentTime, selectedAnnotationId]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = (x / canvas.width) * duration;
    onSeek(Math.max(0, Math.min(duration, time)));
  };

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={40}
      className="w-full h-10 cursor-pointer"
      onClick={handleClick}
    />
  );
} 