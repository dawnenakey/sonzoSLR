import React, { useRef, useEffect, useState } from 'react';
import { Skeleton } from "@/components/ui/skeleton";

export default function VideoThumbnail({ videoUrl, className }) {
  const [thumbnail, setThumbnail] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);
  const canvasRef = useRef(null);
  // videoRef is not used in the current logic, can be removed if not planned for future use
  // const videoRef = useRef(null); 

  useEffect(() => {
    // Reset state for new videoUrl
    setIsLoading(true);
    setError(false);
    setThumbnail('');

    const videoElement = document.createElement('video');
    videoElement.crossOrigin = 'anonymous'; // Important for tainted canvas if video is from different origin
    videoElement.preload = 'metadata'; // We only need metadata and a frame

    const generateThumbnail = () => {
      if (!canvasRef.current) {
        console.error('Canvas ref is not available when trying to generate thumbnail.');
        setError(true);
        setIsLoading(false);
        return;
      }
      try {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');

        if (!context) {
          console.error('Failed to get 2D context from canvas.');
          setError(true);
          setIsLoading(false);
          return;
        }
        
        // Set canvas dimensions to match video
        // Use naturalWidth/Height if video has dimensions, otherwise default or skip
        const videoWidth = videoElement.videoWidth || 300; // Default width if videoWidth is 0
        const videoHeight = videoElement.videoHeight || 150; // Default height if videoHeight is 0
        
        canvas.width = videoWidth;
        canvas.height = videoHeight;
        
        // Draw the current video frame
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert to data URL
        const dataUrl = canvas.toDataURL('image/jpeg');
        setThumbnail(dataUrl);
      } catch (err) {
        console.error('Error generating thumbnail from canvas:', err.message || err);
        setError(true);
      } finally {
        setIsLoading(false);
      }
    };
    
    videoElement.onloadeddata = () => {
      // Seek to 1 second (or a small portion) into the video for a better thumbnail
      // Ensure duration is available before seeking
      if (videoElement.duration && videoElement.duration >= 1) {
        videoElement.currentTime = 1;
      } else if (videoElement.duration && videoElement.duration > 0) {
        videoElement.currentTime = videoElement.duration * 0.1; // e.g. 10% into the video
      } else {
        // If duration is 0 or unavailable, try to generate thumbnail from the first frame immediately
        // This might happen for some video formats or streams before full metadata is loaded
        generateThumbnail();
      }
    };

    videoElement.onseeked = () => {
      generateThumbnail();
    };

    videoElement.onerror = (e) => {
      console.error('Error loading video for thumbnail generation:', e);
      setError(true);
      setIsLoading(false);
    };

    // Handle cases where metadata might not load enough to get duration or dimensions
    // For example, if the video source is invalid or network issue.
    videoElement.onstalled = () => {
        console.warn('Video loading stalled, attempting to generate thumbnail with available data.');
        // Attempt to generate thumbnail even if stalled, might get a black frame or error
        // This is a fallback, ideal case is onseeked or onloadeddata with duration
        if (!thumbnail && isLoading) generateThumbnail(); 
    }
    
    videoElement.src = videoUrl;
    videoElement.load(); // Explicitly call load

    return () => {
      // Cleanup: remove event listeners and revoke object URL if created
      videoElement.onloadeddata = null;
      videoElement.onseeked = null;
      videoElement.onerror = null;
      videoElement.onstalled = null;
      videoElement.src = ''; // Release video resources
      // If we were creating ObjectURLs: URL.revokeObjectURL(videoElement.src);
    };
  }, [videoUrl]); // Rerun effect if videoUrl changes

  if (isLoading) {
    return <Skeleton className={`aspect-video ${className || 'w-full h-auto'}`} />;
  }

  if (error || !thumbnail) { // Show error state if error is true or thumbnail failed to generate
    return (
      <div className={`aspect-video bg-gray-700 flex items-center justify-center ${className || 'w-full h-auto'}`}>
        <span className="text-gray-400 text-xs p-2 text-center">Thumbnail unavailable</span>
      </div>
    );
  }

  return (
    <>
      {/* Canvas is used internally, no need to display it if thumbnail is generated */}
      <canvas ref={canvasRef} className="hidden" /> 
      <img 
        src={thumbnail} 
        alt="Video thumbnail" 
        className={`aspect-video object-cover ${className || 'w-full h-auto'}`}
      />
    </>
  );
}