import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { createPageUrl } from '@/utils';
import { Film, Plus, ArrowRight, Play, Calendar, Clock, Trash2, MoreVertical, Edit2, AlertCircle, UserCircle, Download, Share2, Camera } from 'lucide-react'; // Removed Copy as it wasn't used in the last provided version
import { Button } from '@/components/ui/button';
import { Annotation as AnnotationEntity } from '@/api/entities';
import { Skeleton } from "@/components/ui/skeleton";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu';
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
import { useToast } from "@/components/ui/use-toast";
import ImportVideoDialog from '../components/ImportVideoDialog';
import { Badge } from "@/components/ui/badge";
import { formatTime } from '../components/timeUtils';
import Header from '../components/Header';
import LiveCameraAnnotator from '../components/LiveCameraAnnotator';
import CameraSelector from '../components/CameraSelector';
import { videoAPI } from '@/api/awsClient';

export default function Home() {
  const [videos, setVideos] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isImportDialogOpen, setIsImportDialogOpen] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState(null);
  const [videoToEdit, setVideoToEdit] = useState(null);
  const [apiError, setApiError] = useState(false);
  const { toast } = useToast();
  const [isUploading, setIsUploading] = useState(false);
  const [annotationsMap, setAnnotationsMap] = useState({});
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [cameraSettings, setCameraSettings] = useState({
    resolution: '1920x1080',
    frameRate: 60,
    quality: 'high'
  });

  useEffect(() => {
    fetchVideosAndAnnotations();
  }, []);

  const fetchVideosAndAnnotations = async () => {
    setIsLoading(true);
    setApiError(false);
    try {
      const [fetchedVideos, allAnnotations] = await Promise.all([
        videoAPI.list('-created_date'),
        AnnotationEntity.list()
      ]);

      if (Array.isArray(fetchedVideos)) {
        setVideos(fetchedVideos);
      } else {
        console.error('Fetched videos is not an array:', fetchedVideos);
        setVideos([]);
        setApiError(true);
        toast({
            variant: "destructive",
            title: "Data Error",
            description: "Received invalid video data. Using sample videos."
        });
      }

      if (Array.isArray(allAnnotations)) {
        const annMap = {};
        allAnnotations.forEach(anno => {
          if (!annMap[anno.video_id]) {
            annMap[anno.video_id] = [];
          }
          annMap[anno.video_id].push(anno);
        });
        setAnnotationsMap(annMap);
      } else {
        console.warn('Fetched annotations is not an array:', allAnnotations);
        setAnnotationsMap({});
        if (!Array.isArray(fetchedVideos) || fetchedVideos.length === 0) {
             toast({
                variant: "warning",
                title: "Annotation Data Unavailable",
                description: "Could not load annotation data. Export functionality might be affected."
            });
        }
      }
    } catch (error) {
      console.error('Error fetching videos or annotations:', error);
      setVideos([]);
      setAnnotationsMap({});
      setApiError(true);
      toast({
        variant: "destructive",
        title: "Connection Error",
        description: "Using sample videos. Some features may be limited."
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleVideoImport = async (videoData) => {
    setIsUploading(true);
    try {
      if (apiError) {
        const newVideo = {
          ...videoData,
          id: `fallback-${Date.now()}`,
          created_date: new Date().toISOString()
        };
        setVideos(prev => [newVideo, ...prev].sort((a,b) => new Date(b.created_date) - new Date(a.created_date)));
        toast({
          title: "Video added locally",
          description: "The video was added to your session but won't persist due to connection issues."
        });
        setIsImportDialogOpen(false);
        return;
      }
      
      await videoAPI.create(videoData);
      await fetchVideosAndAnnotations(); // Refresh videos and annotations
      toast({
        title: "Video Imported Successfully",
        description: `"${videoData.title}" is now in your library.`,
      });
      setIsImportDialogOpen(false);
    } catch (error) {
      console.error('Error saving video:', error);
      toast({
        variant: "destructive",
        title: "Error Importing Video",
        description: (error.message && error.message.includes("language")) ? "Please provide the sign language." : "There was a problem importing this video. Please ensure all required fields are filled."
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleEditVideo = async (videoData) => {
    setIsUploading(true);
    try {
      if (apiError) {
        setVideos(prev => 
          prev.map(v => v.id === videoToEdit.id ? {...videoToEdit, ...videoData} : v)
        );
        toast({
          title: "Video updated locally",
          description: "Changes won't persist due to connection issues."
        });
        setVideoToEdit(null);
        setIsImportDialogOpen(false);
        return;
      }
      
      await videoAPI.update(videoToEdit.id, videoData);
      await fetchVideosAndAnnotations(); // Refresh videos and annotations
      toast({
        title: "Video Updated",
        description: `"${videoData.title}" has been updated successfully.`,
      });
      setVideoToEdit(null);
      setIsImportDialogOpen(false);
    } catch (error) {
      console.error('Error updating video:', error);
      toast({
        variant: "destructive",
        title: "Error Updating Video",
        description: (error.message && error.message.includes("language")) ? "Please provide the sign language." : "There was a problem updating this video. Please check the details and try again."
      });
    } finally {
        setIsUploading(false);
    }
  };

  const handleDeleteVideo = async () => {
    if (!videoToDelete) return;
    
    try {
      if (apiError) {
        setVideos(prev => prev.filter(v => v.id !== videoToDelete.id));
        toast({
          title: "Video removed locally",
          description: "The video was removed from your session."
        });
        setVideoToDelete(null);
        return;
      }
      
      // First, fetch all annotations for this video
      const videoAnnotations = await AnnotationEntity.filter({ video_id: videoToDelete.id });
      
      // Delete the video first
      await videoAPI.delete(videoToDelete.id);
      
      // Then delete all associated annotations
      for (const annotation of videoAnnotations) {
        await AnnotationEntity.delete(annotation.id);
      }
      
      await fetchVideosAndAnnotations(); // Refresh videos and annotations
      
      toast({
        title: "Video and Associated Segments Deleted",
        description: `"${videoToDelete.title}" and ${videoAnnotations.length} segments have been removed.`,
      });
    } catch (error) {
      console.error('Error deleting video:', error);
      toast({
        variant: "destructive",
        title: "Error Deleting Video",
        description: "Could not delete the video. Please try again."
      });
    } finally {
      setVideoToDelete(null);
    }
  };

  const formatDuration = (seconds) => {
    if (seconds === null || seconds === undefined || isNaN(seconds)) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  const getThumbnailUrl = (index) => {
    const thumbnails = [
      'https://images.unsplash.com/photo-1592326871020-04f58c1a52f3?q=80&w=500&auto=format&fit=crop',
      'https://images.unsplash.com/photo-1588143335938-dd1282e912d0?q=80&w=500&auto=format&fit=crop',
      'https://images.unsplash.com/photo-1613294326089-d3226eca18ee?q=80&w=500&auto=format&fit=crop',
      'https://images.unsplash.com/photo-1614641917479-312ff3f95a60?q=80&w=500&auto=format&fit=crop',
      'https://images.unsplash.com/photo-1596524430615-b46475ddff6e?q=80&w=500&auto=format&fit=crop'
    ];
    return thumbnails[index % thumbnails.length];
  };
  
  const getVideoColor = (title) => {
    const colorClasses = [
      'bg-blue-500', 'bg-purple-500', 'bg-rose-500', 'bg-amber-500', 'bg-emerald-500', 'bg-indigo-500'
    ];
    const titleHash = [...(title || "")].reduce((hash, char) => char.charCodeAt(0) + hash, 0);
    return colorClasses[titleHash % colorClasses.length];
  };

  const handleDownloadVideo = (videoUrl, videoTitle) => {
    if (!videoUrl) {
        toast({ variant: "destructive", title: "Download Error", description: "Video URL is not available." });
        return;
    }
    try {
        const link = document.createElement('a');
        link.href = videoUrl;
        const filename = videoUrl.substring(videoUrl.lastIndexOf('/') + 1).split('?')[0] || `${videoTitle.replace(/\s+/g, '_')}.mp4`;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        toast({ title: "Download Started", description: `Downloading "${videoTitle}"...`});
    } catch (error) {
        console.error("Error downloading video:", error);
        toast({ variant: "destructive", title: "Download Error", description: "Could not start video download." });
    }
  };

  const handleShareVideo = (videoId, videoTitle) => {
    const shareUrl = `${window.location.origin}${createPageUrl(`Annotator?id=${videoId}`)}`;
    navigator.clipboard.writeText(shareUrl).then(() => {
        toast({ title: "Link Copied!", description: `Shareable link for "${videoTitle}" copied to clipboard.` });
    }).catch(err => {
        console.error("Error copying share link:", err);
        toast({ variant: "destructive", title: "Copy Error", description: "Could not copy share link."});
    });
  };

  const handleExportAnnotations = (videoId, videoTitle) => {
    const videoAnnotations = annotationsMap[videoId] || [];
    
    if (apiError && videoAnnotations.length === 0) {
        toast({
            title: "Export Unavailable",
            description: "Cannot export annotations due to a connection error.",
            variant: "warning"
        });
        return;
    }

    if (videoAnnotations.length === 0) {
      toast({
        title: "No Annotations",
        description: `There are no annotations to export for "${videoTitle}".`,
        variant: "info"
      });
      return;
    }
  
    const formattedAnnotations = videoAnnotations.map(annotation => ({
      segment_type: annotation.segment_type,
      start_time: formatTime(annotation.start_time),
      duration: annotation.duration.toFixed(2),
      description: annotation.description || ""
    }));
  
    const jsonString = JSON.stringify(formattedAnnotations, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${videoTitle.replace(/\s+/g, '_')}_Annotations.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast({
      title: "Export Started",
      description: `Annotations for "${videoTitle}" are being downloaded.`
    });
  };

  return (
    <>
      <Header />
      <div className="max-w-7xl mx-auto p-4 sm:p-6">
        {/* Camera Selection Section */}
        <section className="mb-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <CameraSelector 
                onCameraSelect={setSelectedCamera}
                onSettingsChange={setCameraSettings}
                showSettings={true}
              />
            </div>
            <div className="lg:col-span-2">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <Camera className="h-5 w-5 text-blue-600" />
                  Live Camera Recording
                  {selectedCamera?.isBRIO && (
                    <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full font-medium">
                      BRIO Optimized
                    </span>
                  )}
                </h2>
                <LiveCameraAnnotator onVideoUploaded={async (file) => {
                  // Simulate ImportVideoDialog's upload flow: open dialog with file pre-selected
                  setIsImportDialogOpen(true);
                  // Optionally, you could auto-fill ImportVideoDialog with the file, or handle upload directly here
                  // For now, user will fill in details in the dialog after upload
                }} />
              </div>
            </div>
          </div>
        </section>
        
        <section>
          <div className="flex flex-col sm:flex-row justify-between items-center mb-6 gap-4">
            <h2 className="text-2xl font-bold text-gray-900">Your Video Library</h2>
            <Button 
              variant="outline" 
              onClick={() => { setVideoToEdit(null); setIsImportDialogOpen(true); }}
              className="w-full sm:w-auto"
            >
              <Plus className="mr-2 h-4 w-4" />
              Import Video
            </Button>
          </div>
        </section>

        {isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
              {[1, 2, 3].map((i) => (
                <div key={i} className="bg-white rounded-xl overflow-hidden shadow-lg">
                  <Skeleton className="h-40 w-full" />
                  <div className="p-4">
                    <Skeleton className="h-6 w-3/4 mb-2" />
                    <Skeleton className="h-4 w-1/2 mb-3" />
                    <Skeleton className="h-8 w-1/3 ml-auto" />
                  </div>
                </div>
              ))}
            </div>
          ) : videos.length === 0 ? (
            <div className="bg-white rounded-xl p-6 sm:p-8 text-center border-2 border-dashed border-gray-200">
              {/* ASL Welcome Video */}
              <div className="mx-auto w-64 h-36 mb-4">
                <video
                  src="/welcome-asl.mov"
                  className="w-full h-full rounded-lg shadow"
                  autoPlay
                  loop
                  muted
                  playsInline
                  aria-label="Welcome message in ASL"
                />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No videos yet</h3>
              <p className="text-gray-500 mb-6 max-w-md mx-auto">
                Import your first sign language video to start segmenting and annotating.
              </p>
              <Button onClick={() => { setVideoToEdit(null); setIsImportDialogOpen(true); }} size="lg">
                <Plus className="mr-2 h-4 w-4" />
                Import Video
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
              {videos.map((video, index) => {
                const hasValidId = video && video.id && String(video.id).trim() !== "";
                const annotatorUrl = hasValidId ? createPageUrl(`Annotator?id=${video.id}`) : "#";
                const videoHasAnnotations = (annotationsMap[video.id] || []).length > 0;

                return (
                <div key={video.id || `no-id-${index}`} className="bg-white rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 group flex flex-col relative">
                  <div className="absolute top-2 right-2 z-10">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-8 w-8 bg-black/50 hover:bg-black/70 text-white rounded-full"
                        >
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem 
                          className="cursor-pointer flex items-center gap-2"
                          onClick={() => { setVideoToEdit(video); setIsImportDialogOpen(true); }}
                          disabled={!hasValidId || apiError}
                        >
                          <Edit2 className="h-4 w-4" />
                          Edit Details
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          className="cursor-pointer flex items-center gap-2"
                          onClick={() => handleDownloadVideo(video.url, video.title)}
                          disabled={!video.url || apiError}
                        >
                          <Download className="h-4 w-4" />
                          Download Video
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          className="cursor-pointer flex items-center gap-2"
                          onClick={() => handleExportAnnotations(video.id, video.title)}
                          disabled={!videoHasAnnotations || !hasValidId || apiError}
                        >
                          <Download className="h-4 w-4" />
                          Export JSON
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          className="cursor-pointer flex items-center gap-2"
                          onClick={() => hasValidId && handleShareVideo(video.id, video.title)}
                          disabled={!hasValidId || apiError}
                        >
                          <Share2 className="h-4 w-4" />
                          Share
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem 
                          className="text-red-600 hover:!text-red-700 hover:!bg-red-50 cursor-pointer flex items-center gap-2"
                          onClick={() => setVideoToDelete(video)}
                          disabled={!hasValidId || apiError}
                        >
                          <Trash2 className="h-4 w-4" />
                          Delete Video
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                
                  <Link
                    to={annotatorUrl}
                    className={`aspect-video relative overflow-hidden block ${!hasValidId ? 'pointer-events-none opacity-70' : ''}`}
                    onClick={(e) => !hasValidId && e.preventDefault()}
                    aria-disabled={!hasValidId}
                    tabIndex={!hasValidId ? -1 : undefined}
                  >
                    {video.thumbnail_url ? (
                      <img 
                        src={video.thumbnail_url} 
                        alt={video.title} 
                        className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                      />
                    ) : index < 5 ? ( 
                      <img 
                        src={getThumbnailUrl(index)} 
                        alt={video.title} 
                        className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                      />
                    ) : (
                      <div className={`w-full h-full flex items-center justify-center ${getVideoColor(video.title)}`}>
                        <Film className="h-16 w-16 text-white/90" />
                      </div>
                    )}
                    
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/30 to-transparent opacity-80 group-hover:opacity-70 transition-opacity"></div>
                    
                    {hasValidId && (
                      <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <div className="bg-white/90 rounded-full p-3 shadow-lg transform group-hover:scale-110 transition-transform">
                          <Play className="h-8 w-8 text-indigo-600 fill-indigo-600" />
                        </div>
                      </div>
                    )}
                    
                    <div className="absolute top-3 left-3 flex gap-2">
                       {video.language && (
                         <Badge className="bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded-sm truncate max-w-[100px]" title={video.language}>
                           {video.language}
                         </Badge>
                       )}
                      <div className="bg-black/70 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1" title={formatDuration(video.duration) === 'N/A' ? 'Duration is not available' : undefined}>
                        <Clock className="h-3 w-3" />
                        {formatDuration(video.duration)}
                      </div>
                    </div>
                    
                    <div className="absolute bottom-0 left-0 right-0 p-4">
                      <h3 className="font-bold text-white text-lg mb-1 drop-shadow-sm truncate" title={video.title}>
                        {video.title}
                      </h3>
                      <div className="flex items-center gap-2 mb-1 text-xs">
                        <span className="text-white/80 flex items-center">
                          <Calendar className="h-3 w-3 mr-1" />
                          {new Date(video.created_date).toLocaleDateString()}
                        </span>
                         {video.author_name && (
                          <span className="text-white/80 flex items-center truncate" title={`By: ${video.author_name}`}>
                            <UserCircle className="h-3 w-3 mr-1" /> By: {video.author_name}
                          </span>
                         )}
                      </div>
                    </div>
                  </Link>
                
                  <div className="p-4 flex flex-col flex-grow">
                    <p className="text-sm text-gray-600 mb-3 line-clamp-2 flex-grow">
                      {video.description || 'No description available.'}
                    </p>
                    <div className="mt-auto pt-2 border-t border-gray-100">
                      <Link 
                        to={annotatorUrl} 
                        className={`w-full ${!hasValidId ? 'pointer-events-none' : ''}`}
                        onClick={(e) => !hasValidId && e.preventDefault()}
                        aria-disabled={!hasValidId}
                        tabIndex={!hasValidId ? -1 : undefined}
                      >
                        <Button 
                          variant="ghost" 
                          className="w-full text-indigo-600 hover:bg-indigo-50 justify-center"
                          disabled={!hasValidId}
                        >
                          Annotate <ArrowRight className="ml-2 h-4 w-4" />
                        </Button>
                      </Link>
                      {!hasValidId && (
                        <p className="text-xs text-red-500 text-center mt-1">Video ID missing, cannot annotate.</p>
                      )}
                    </div>
                  </div>
                </div>
              )})}
            </div>
          )}

        <ImportVideoDialog 
          isOpen={isImportDialogOpen}
          onClose={() => {
            setIsImportDialogOpen(false);
            setVideoToEdit(null); 
          }}
          onVideoImport={videoToEdit ? handleEditVideo : handleVideoImport}
          editVideo={videoToEdit}
        />
        
         <AlertDialog open={!!videoToDelete} onOpenChange={() => setVideoToDelete(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Are you sure you want to delete this video?</AlertDialogTitle>
              <AlertDialogDescription>
                This will permanently delete "{videoToDelete?.title}" and all its annotations. 
                This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                className="bg-red-600 hover:bg-red-700"
                onClick={handleDeleteVideo}
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </>
  );
}
