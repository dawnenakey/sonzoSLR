import { useState } from "react";
import "./App.css";
import { ImportVideoDialog } from "./components/ImportVideoDialog";
import { Button } from "./components/ui/button";
import AnnotationControls from "./components/AnnotationControls";
import AnnotationList from "./components/AnnotationList";
import AnnotationTimeline from "./components/AnnotationTimeline";
import VideoPlayer from "./components/VideoPlayer";
import { base44, signIn, signOut, getAuthStatus, initializeBase44, User } from './api/base44Client';

// Assuming these types are defined or imported somewhere appropriate
interface Annotation {
  id: string;
  startTime: number;
  endTime: number;
  label: string;
  notes?: string;
  video_id: string;
}

interface Video {
  id: string;
  name: string;
  url: string;
  annotations: Annotation[];
}

function App() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isSegmenting, setIsSegmenting] = useState(false);
  
  // Define the base API URL
  const apiUrl = "https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod";

  const handleFileSelect = (file: File) => {
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
  };

  const handleAcceptAndUpload = async () => {
    if (!selectedFile) {
      alert("No file selected!");
      return;
    }

    setIsUploading(true);
    try {
      const sessionApiUrl = `${apiUrl}/sessions`;
      const sessionResponse = await fetch(sessionApiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: selectedFile.name,
          description: "Video uploaded from annotation tool",
        }),
      });

      if (!sessionResponse.ok) throw new Error(`Failed to create session: ${sessionResponse.status}`);
      const sessionData = await sessionResponse.json();
      const { session_id } = sessionData;

      const presignApiUrl = `${apiUrl}/sessions/${session_id}/upload-video`;
      const presignResponse = await fetch(presignApiUrl, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          filename: selectedFile.name,
          contentType: selectedFile.type
        })
      });

      if (!presignResponse.ok) throw new Error("Failed to get pre-signed URL");
      const { uploadUrl } = await presignResponse.json();

      if (!uploadUrl) throw new Error("Pre-signed URL not found in the response.");

      const uploadResponse = await fetch(uploadUrl, {
        method: "PUT",
        body: selectedFile,
        headers: { 'Content-Type': selectedFile.type }
      });

      if (!uploadResponse.ok) throw new Error(`S3 upload failed with status: ${uploadResponse.status}`);

      alert('File uploaded successfully!');
      handleReject(); // Reset UI after successful upload

    } catch (error) {
      console.error("Upload failed:", error);
      alert(`Upload failed: ${error}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>SpokHand SLR Annotation Tool</h1>
      </header>
      <main className="p-4">
        <div className="video-container mx-auto bg-black">
          {videoUrl && <VideoPlayer src={videoUrl} onTimeUpdate={setCurrentTime} onDurationChange={setVideoDuration} />}
        </div>

        <div className="controls p-4 my-4 border rounded-md">
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
                    {isUploading ? 'Uploading...' : 'Accept & Upload'}
                  </Button>
              </div>
            )}
        </div>
        
        {videoUrl && (
            <div className="annotation-section grid grid-cols-3 gap-8">
              <div className="col-span-2 space-y-4">
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
                />
              </div>

              <div className="col-span-1">
                <AnnotationList
                    annotations={annotations}
                    onEdit={() => {}}
                    onDelete={() => {}}
                    onSelect={() => {}}
                />
              </div>
            </div>
        )}

      </main>
    </div>
  );
}

export default App;
