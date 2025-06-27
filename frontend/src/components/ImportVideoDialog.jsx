import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Upload, AlertCircle, Image, Camera, Paintbrush, Languages, UserCircle, Clock } from 'lucide-react'; // Added Clock

export default function ImportVideoDialog({ isOpen, onClose, onVideoImport, editVideo = null }) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [duration, setDuration] = useState(''); // Store as string for input, convert to number on save
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [thumbnailTab, setThumbnailTab] = useState("auto");
  const [thumbnailFile, setThumbnailFile] = useState(null);
  const [thumbnailUrl, setThumbnailUrl] = useState('');
  const [thumbnailPrompt, setThumbnailPrompt] = useState('Sign language interpreter against a blue background');
  const [isGeneratingThumbnail, setIsGeneratingThumbnail] = useState(false);
  const [generatedThumbnailUrl, setGeneratedThumbnailUrl] = useState('');
  const [language, setLanguage] = useState('');
  const [authorName, setAuthorName] = useState('');
  
  const fileInputRef = useRef(null);
  const thumbnailInputRef = useRef(null);
  const videoMetadataRef = useRef(null); // For hidden video element

  const resetForm = () => {
    setTitle('');
    setDescription('');
    setFile(null);
    setVideoUrl('');
    setDuration('');
    setThumbnailTab("auto");
    setThumbnailFile(null);
    setThumbnailUrl('');
    setGeneratedThumbnailUrl('');
    setLanguage('');
    setAuthorName('');
    setError('');
    if (videoMetadataRef.current) {
        URL.revokeObjectURL(videoMetadataRef.current.src); // Clean up previous object URLs
        videoMetadataRef.current.removeAttribute('src');
        videoMetadataRef.current.load();
    }
  };

  useEffect(() => {
    if (isOpen) {
        resetForm();
        if (editVideo) {
            setTitle(editVideo.title || '');
            setDescription(editVideo.description || '');
            setVideoUrl(editVideo.url || '');
            setDuration(editVideo.duration !== null && editVideo.duration !== undefined ? String(editVideo.duration) : '');
            setLanguage(editVideo.language || '');
            setAuthorName(editVideo.author_name || '');
            if (editVideo.thumbnail_url) {
                setThumbnailUrl(editVideo.thumbnail_url);
                setThumbnailTab('upload'); 
            } else {
                 setThumbnailTab('auto');
                 setThumbnailUrl('');
            }
            if (editVideo.url) {
                getVideoDurationFromSource(editVideo.url, 'url');
            }
        }
    }
  }, [editVideo, isOpen]);


  const getVideoDurationFromSource = (source, sourceType) => {
    if (!videoMetadataRef.current) return;
    const videoElement = videoMetadataRef.current;

    const handleLoadedMetadata = () => {
        if (videoElement.duration && isFinite(videoElement.duration)) {
            setDuration(String(Math.round(videoElement.duration)));
        }
        // Clean up event listeners
        videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
        videoElement.removeEventListener('error', handleErrorLoadingMetadata);
        if (sourceType === 'file') {
            URL.revokeObjectURL(videoElement.src); // Clean up blob URL after use
        }
    };

    const handleErrorLoadingMetadata = (e) => {
        console.warn('Error loading video metadata for duration:', e);
        // Optionally set error or keep duration as is (manual input)
        videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
        videoElement.removeEventListener('error', handleErrorLoadingMetadata);
        if (sourceType === 'file') {
            URL.revokeObjectURL(videoElement.src);
        }
    };
    
    videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);
    videoElement.addEventListener('error', handleErrorLoadingMetadata);

    if (sourceType === 'file' && source instanceof File) {
        videoElement.src = URL.createObjectURL(source);
    } else if (sourceType === 'url' && typeof source === 'string') {
        videoElement.src = source;
    }
    videoElement.load(); // Start loading the video
  };


  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type.startsWith('video/')) {
        setFile(selectedFile);
        setVideoUrl(''); // Clear URL if file is selected
        setError('');
        getVideoDurationFromSource(selectedFile, 'file');
      } else {
        setError('Please select a valid video file.');
        setFile(null);
      }
    }
  };

  const handleVideoUrlChange = (e) => {
    const url = e.target.value;
    setVideoUrl(url);
    if (url) {
        setFile(null); // Clear file if URL is entered
        getVideoDurationFromSource(url, 'url');
    } else {
        setDuration(''); // Clear duration if URL is cleared
    }
  };


  const handleThumbnailFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type.startsWith('image/')) {
        setThumbnailFile(selectedFile);
        setThumbnailUrl(URL.createObjectURL(selectedFile)); 
        setError('');
      } else {
        setError('Please select a valid image file.');
        setThumbnailFile(null);
      }
    }
  };

  const handleSubmit = async () => {
    if (!title.trim()) {
      setError("Please enter a title for the video.");
      return;
    }
    if (!authorName.trim()) {
      setError("Please enter the author's name.");
      return;
    }
    if (!editVideo && !videoUrl && !file) { 
      setError("Please select a video file or provide a URL.");
      return;
    }
    if (editVideo && !videoUrl) { 
        setError("Video URL is missing for existing video.");
        return;
    }
    const parsedDuration = parseFloat(duration);
    if (duration.trim() !== '' && (isNaN(parsedDuration) || parsedDuration < 0)) {
        setError("Please enter a valid positive number for duration, or leave it blank.");
        return;
    }


    try {
      setIsUploading(true);
      setError('');
      let finalVideoUrl = videoUrl;
      let finalThumbnailUrl = thumbnailUrl; 
      
      if (file) { 
        // TODO: Implement file upload
        throw new Error("File upload functionality not implemented.");
      }
      
      if (thumbnailTab === "upload") {
        if (thumbnailFile) { 
          // TODO: Implement custom thumbnail upload
          throw new Error("Custom thumbnail upload functionality not implemented.");
        } else if (editVideo && editVideo.thumbnail_url && thumbnailUrl === editVideo.thumbnail_url) {
            finalThumbnailUrl = editVideo.thumbnail_url;
        }
      } else if (thumbnailTab === "ai" && generatedThumbnailUrl) {
        finalThumbnailUrl = generatedThumbnailUrl;
      } else if (thumbnailTab === "auto") {
        finalThumbnailUrl = ''; 
      }
      
      const videoData = {
        title,
        description,
        url: finalVideoUrl,
        thumbnail_url: finalThumbnailUrl,
        language: language.trim(),
        author_name: authorName,
        duration: duration.trim() === '' ? null : parsedDuration, // Save duration as number or null
      };
      
      await onVideoImport(videoData);
    } catch (err) {
      console.error("Error processing video:", err);
      setError(`Error: ${err.message || "Unknown error during processing."}`);
    } finally {
      setIsUploading(false);
    }
  };

  const generateAIThumbnail = async () => {
    if (!thumbnailPrompt.trim()) {
      setError("Please enter a description for the thumbnail.");
      return;
    }
    try {
      setIsGeneratingThumbnail(true);
      setError('');
      // TODO: Implement image generation
      throw new Error("Image generation functionality not implemented.");
    } catch (err) {
      console.error("Error generating thumbnail:", err);
      setError(`Error generating thumbnail: ${err.message || "Unknown error"}`);
    } finally {
      setIsGeneratingThumbnail(false);
    }
  };
  
  const currentDialogTitle = editVideo ? 'Edit Video Details' : 'Import New Video';
  const currentDialogDescription = editVideo 
    ? 'Update the details for this video.' 
    : 'Upload a video file or provide a URL, and specify its details.';

  return (
    <Dialog open={isOpen} onOpenChange={(openState) => {
      if (!openState) {
        onClose();
      }
    }}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Hidden video element for metadata fetching */}
        <video ref={videoMetadataRef} style={{ display: 'none' }} crossOrigin="anonymous" preload="metadata"></video>
        <DialogHeader>
          <DialogTitle>{currentDialogTitle}</DialogTitle>
          <DialogDescription>{currentDialogDescription}</DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6 py-4">
          {error && (
            <div className="bg-destructive/15 text-destructive text-sm p-3 rounded-md flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="title">Title <span className="text-red-500">*</span></Label>
            <Input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter video title"
              disabled={isUploading}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional: A brief summary of the video's content"
              disabled={isUploading}
              rows={3}
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="language">
                Sign Language
                <div className="text-xs text-muted-foreground mt-0.5">E.g., ASL, LSF, BSL. (Optional)</div>
              </Label>
              <div className="relative">
                <Input
                  id="language"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  placeholder="Enter sign language"
                  disabled={isUploading}
                  className="pl-9"
                />
                <Languages className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="authorName">
                Author Name <span className="text-red-500">*</span>
                <div className="text-xs text-muted-foreground mt-0.5">Full name of the video creator or uploader.</div>
              </Label>
              <div className="relative">
                <Input
                  id="authorName"
                  value={authorName}
                  onChange={(e) => setAuthorName(e.target.value)}
                  placeholder="E.g., Jane Doe"
                  disabled={isUploading}
                  className="pl-9"
                />
                <UserCircle className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              </div>
            </div>
          </div>

          {!editVideo && ( 
            <div className="space-y-2">
              <Label>Video Source <span className="text-red-500">*</span></Label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
                <div>
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full h-20 flex flex-col items-center justify-center gap-1.5 border-dashed hover:border-primary"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                  >
                    <Upload className="h-5 w-5 text-muted-foreground" />
                    <span className="text-sm font-normal">Upload Video File</span>
                  </Button>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="video/*"
                    className="hidden"
                  />
                  {file && (
                    <div className="mt-1.5 text-xs text-muted-foreground truncate">
                      Selected: {file.name}
                    </div>
                  )}
                </div>
                
                <div className="relative">
                   <Input
                    id="videoUrl"
                    type="url"
                    placeholder="Or paste video URL (e.g., YouTube, Vimeo)"
                    value={videoUrl}
                    onChange={handleVideoUrlChange}
                    disabled={isUploading || !!file}
                    className="h-20 pl-9 text-sm" 
                  />
                  <Languages className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                   {videoUrl && !file && (
                    <div className="mt-1.5 text-xs text-muted-foreground">
                      Using provided URL.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
           {editVideo && ( 
            <div className="space-y-2">
                <Label htmlFor="currentVideoUrl">Current Video URL</Label>
                <Input
                    id="currentVideoUrl"
                    type="text"
                    value={videoUrl}
                    readOnly
                    disabled
                    className="bg-muted/50"
                />
                 <p className="text-xs text-muted-foreground">Video URL cannot be changed after import. To use a different video, import it as a new entry.</p>
            </div>
            )}

            <div className="space-y-2">
                <Label htmlFor="duration">
                    Duration (seconds)
                    <div className="text-xs text-muted-foreground mt-0.5">Auto-detected if possible. You can override it.</div>
                </Label>
                <div className="relative">
                <Input
                    id="duration"
                    type="number"
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    placeholder="E.g., 120 (for 2 minutes)"
                    disabled={isUploading}
                    min="0"
                    step="any"
                    className="pl-9"
                />
                <Clock className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                </div>
            </div>
          
          <div className="space-y-2">
            <Label>Video Thumbnail (Optional)</Label>
            <Tabs value={thumbnailTab} onValueChange={setThumbnailTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="auto" disabled={isUploading} className="text-xs sm:text-sm">
                  <Image className="h-3.5 w-3.5 mr-1.5 sm:h-4 sm:w-4" /> Auto
                </TabsTrigger>
                <TabsTrigger value="upload" disabled={isUploading} className="text-xs sm:text-sm">
                  <Camera className="h-3.5 w-3.5 mr-1.5 sm:h-4 sm:w-4" /> Upload
                </TabsTrigger>
                <TabsTrigger value="ai" disabled={isUploading} className="text-xs sm:text-sm">
                   <Paintbrush className="h-3.5 w-3.5 mr-1.5 sm:h-4 sm:w-4" /> AI Gen
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="auto" className="p-3 border rounded-b-md min-h-[80px] flex items-center justify-center">
                <p className="text-sm text-muted-foreground text-center">
                  A thumbnail will be automatically generated from the video if possible.
                </p>
              </TabsContent>
              
              <TabsContent value="upload" className="p-3 border rounded-b-md min-h-[80px]">
                <Button
                  type="button"
                  variant="outline"
                  className="w-full h-20 flex flex-col items-center justify-center gap-1.5 border-dashed hover:border-primary"
                  onClick={() => thumbnailInputRef.current?.click()}
                  disabled={isUploading}
                >
                  {(thumbnailUrl && !thumbnailFile) ? ( 
                    <img src={thumbnailUrl} alt="Current thumbnail" className="max-h-full max-w-full object-contain rounded" />
                  ) : (thumbnailFile) ? (
                    <img src={URL.createObjectURL(thumbnailFile)} alt="New thumbnail preview" className="max-h-full max-w-full object-contain rounded" />
                  ) : (
                     <>
                        <Upload className="h-5 w-5 text-muted-foreground" />
                        <span className="text-sm font-normal">Upload Custom Thumbnail</span>
                     </>
                  )}
                </Button>
                <input
                  type="file"
                  ref={thumbnailInputRef}
                  onChange={handleThumbnailFileChange}
                  accept="image/*"
                  className="hidden"
                />
                {thumbnailFile && (
                  <div className="mt-1.5 text-xs text-muted-foreground truncate">
                    New: {thumbnailFile.name}
                  </div>
                )}
                 {!thumbnailFile && thumbnailUrl && !thumbnailUrl.startsWith('blob:') && editVideo && (
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    Current thumbnail is active. Upload a new one to change.
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="ai" className="p-3 border rounded-b-md min-h-[80px]">
                <div className="space-y-3">
                  <div>
                    <Input
                      value={thumbnailPrompt}
                      onChange={(e) => setThumbnailPrompt(e.target.value)}
                      placeholder="Describe the thumbnail image (e.g., person signing 'hello')"
                      disabled={isGeneratingThumbnail || isUploading}
                      className="text-sm"
                    />
                  </div>
                  <Button 
                    type="button"
                    onClick={generateAIThumbnail}
                    disabled={isGeneratingThumbnail || isUploading || !thumbnailPrompt.trim()}
                    className="w-full text-sm"
                  >
                    {isGeneratingThumbnail ? (
                      <><div className="animate-spin rounded-full h-3.5 w-3.5 border-2 border-t-transparent border-current mr-2"></div>Generating...</>
                    ) : (
                      <>Generate with AI</>
                    )}
                  </Button>
                  
                  {generatedThumbnailUrl && (
                    <div className="mt-3 border rounded-md overflow-hidden aspect-video">
                      <img 
                        src={generatedThumbnailUrl} 
                        alt="AI Generated thumbnail" 
                        className="w-full h-full object-cover"
                      />
                    </div>
                  )}
                   {!generatedThumbnailUrl && thumbnailUrl && thumbnailTab === 'ai' && editVideo && (
                     <div className="mt-3 border rounded-md overflow-hidden aspect-video">
                      <img 
                        src={thumbnailUrl} 
                        alt="Current AI thumbnail" 
                        className="w-full h-full object-cover"
                      />
                    </div>
                   )}
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
        
        <DialogFooter className="mt-2">
          <Button 
            variant="outline" 
            onClick={() => {
              onClose();
            }}
            disabled={isUploading}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleSubmit}
            disabled={isUploading}
          >
            {isUploading ? (
              <><div className="animate-spin rounded-full h-4 w-4 border-2 border-t-transparent border-current mr-2"></div>{editVideo ? "Saving..." : "Importing..."}</>
            ) : (
              <>{editVideo ? "Save Changes" : "Import Video"}</>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}