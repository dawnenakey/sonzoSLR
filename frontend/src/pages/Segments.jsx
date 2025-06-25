
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { createPageUrl } from '@/utils';
import { 
  Search, 
  SlidersHorizontal, 
  Film, 
  ArrowLeft, 
  ArrowRight, 
  Clock,
  Check, 
  HandMetal, 
  Pause as PauseIcon, 
  CornerUpRight, 
  AlertCircle, 
  MoreVertical, 
  Trash2, 
  Share2 
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { Annotation as AnnotationEntity } from '@/api/entities';
import { Video as VideoEntity } from '@/api/entities';
import { formatTime } from '../components/timeUtils';
import AnnotationDetailDialog from '../components/AnnotationDetailDialog';

export default function SegmentsPage() {
  const [annotations, setAnnotations] = useState([]);
  const [videos, setVideos] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [videoFilter, setVideoFilter] = useState('all');
  const [qualifiedFilter, setQualifiedFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('newest');
  const [selectedAnnotation, setSelectedAnnotation] = useState(null);
  const [isDetailDialogOpen, setIsDetailDialogOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [expandedFilters, setExpandedFilters] = useState(true); 
  const [loadingErrors, setLoadingErrors] = useState([]);
  const [annotationToDelete, setAnnotationToDelete] = useState(null);
  
  const [hoveredSegmentId, setHoveredSegmentId] = useState(null);
  const [isCleaningUp, setIsCleaningUp] = useState(false);
  
  const { toast } = useToast();
  const ITEMS_PER_PAGE = 20;
  
  useEffect(() => {
    fetchDataAndCleanup();
  }, []);

  const fetchDataAndCleanup = async () => {
    setIsLoading(true);
    setIsCleaningUp(true);
    const errors = [];
    
    try {
      let allVideos;
      try {
        allVideos = await VideoEntity.list();
      } catch (error) {
        console.error("Error fetching videos:", error);
        toast({
          variant: "destructive",
          title: "Error Loading Videos",
          description: "Could not verify video library state. Some segments might be unavailable."
        });
        setIsLoading(false);
        setIsCleaningUp(false);
        return;
      }
      
      const validVideoIds = new Set(allVideos.map(v => v.id));
      const allAnnotations = await AnnotationEntity.list();
      const orphanedSegments = allAnnotations.filter(a => !validVideoIds.has(a.video_id));
      
      if (orphanedSegments.length > 0) {
        for (const segment of orphanedSegments) {
          try {
            await AnnotationEntity.delete(segment.id);
          } catch (error) {
            console.warn(`Failed to delete orphaned segment ${segment.id}:`, error);
            errors.push(`Segment ${segment.id}`);
          }
        }
        toast({
          title: "Database Cleaned",
          description: `Removed ${orphanedSegments.length} orphaned segments.${errors.length ? ' Some deletions failed.' : ''}`,
          duration: 5000,
        });
      }
      
      const validAnnotations = allAnnotations.filter(a => validVideoIds.has(a.video_id)).sort((a,b) => new Date(b.created_date) - new Date(a.created_date));
      setAnnotations(validAnnotations);
      
      const videoMap = {};
      for (const video of allVideos) {
        videoMap[video.id] = video;
      }
      setVideos(videoMap);
      
    } catch (error) {
      console.error("Error during data cleanup:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "There was a problem cleaning up the segment database.",
      });
    } finally {
      setIsLoading(false);
      setIsCleaningUp(false);
    }
  };
  
  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
    setCurrentPage(1); 
  };
  
  const handleAnnotationClick = (annotation) => {
    setSelectedAnnotation(annotation);
    setIsDetailDialogOpen(true);
  };
  
  const handleAnnotationUpdate = async (updatedAnnotation) => {
    try {
      await AnnotationEntity.update(updatedAnnotation.id, updatedAnnotation);
      setAnnotations(prev => 
        prev.map(a => a.id === updatedAnnotation.id ? updatedAnnotation : a)
      );
      toast({
        title: "Segment updated",
        description: "Changes saved successfully.",
      });
    } catch (error) {
      console.error("Error updating annotation:", error);
      toast({
        variant: "destructive",
        title: "Update failed",
        description: "Could not save changes to the segment.",
      });
    }
  };
  
  const confirmDeleteAnnotation = (annotation) => {
    setAnnotationToDelete(annotation);
  };

  const executeDeleteAnnotation = async () => {
    if (!annotationToDelete) return;
    try {
      await AnnotationEntity.delete(annotationToDelete.id);
      setAnnotations(prev => prev.filter(a => a.id !== annotationToDelete.id));
      toast({
        title: "Segment deleted",
        description: "The segment has been removed.",
      });
    } catch (error) {
      console.error("Error deleting annotation:", error);
      toast({
        variant: "destructive",
        title: "Delete failed",
        description: "Could not remove the segment.",
      });
    } finally {
      setAnnotationToDelete(null);
    }
  };
  
  const handleShareSegment = (annotation) => {
    const shareUrl = `${window.location.origin}${createPageUrl(`Segments?id=${annotation.id}`)}`;
    navigator.clipboard.writeText(shareUrl).then(() => {
        toast({ title: "Link Copied!", description: `Shareable link for segment copied to clipboard.` });
    }).catch(err => {
        console.error("Error copying share link:", err);
        toast({ variant: "destructive", title: "Copy Error", description: "Could not copy share link."});
    });
  };

  const getSegmentIcon = (type) => {
    switch (type) {
      case 'SIGN_UNIT': return <HandMetal className="h-4 w-4" />;
      case 'BREAK': return <PauseIcon className="h-4 w-4" />;
      case 'TRANSITION': return <CornerUpRight className="h-4 w-4" />;
      case 'FALSE_POSITIVE': return <AlertCircle className="h-4 w-4" />;
      default: return <Film className="h-4 w-4" />;
    }
  };

  const filteredAndSortedAnnotations = annotations.filter(annotation => {
    const videoOfAnnotation = videos[annotation.video_id];
    const videoTitle = videoOfAnnotation ? videoOfAnnotation.title : 'missing video';

    const matchesSearch = searchQuery === '' || 
      (annotation.description?.toLowerCase().includes(searchQuery.toLowerCase()) || 
       annotation.label?.toLowerCase().includes(searchQuery.toLowerCase()) ||
       videoTitle.toLowerCase().includes(searchQuery.toLowerCase()));
       
    const matchesType = typeFilter === 'all' || annotation.segment_type === typeFilter;
    
    const matchesVideo = videoFilter === 'all' || annotation.video_id === videoFilter;
    
    const matchesQualified = qualifiedFilter === 'all' || 
      (qualifiedFilter === 'qualified' && annotation.qualified) || 
      (qualifiedFilter === 'unqualified' && !annotation.qualified);
      
    return matchesSearch && matchesType && matchesVideo && matchesQualified;
  }).sort((a, b) => {
    switch(sortOrder) {
      case 'newest':
        return new Date(b.created_date) - new Date(a.created_date);
      case 'oldest':
        return new Date(a.created_date) - new Date(b.created_date);
      case 'duration_asc':
        return a.duration - b.duration;
      case 'duration_desc':
        return b.duration - a.duration;
      default:
        return 0;
    }
  });
  
  const totalPages = Math.ceil(filteredAndSortedAnnotations.length / ITEMS_PER_PAGE);
  const paginatedAnnotations = filteredAndSortedAnnotations.slice(
    (currentPage - 1) * ITEMS_PER_PAGE, 
    currentPage * ITEMS_PER_PAGE
  );
  
  const videoOptions = Object.values(videos)
    .filter(v => !v.missing) 
    .map(v => ({ id: v.id, title: v.title }));

  // Function to handle video element initialization and positioning
  const setupVideoSegmentView = (videoElement, annotation) => {
    if (!videoElement || !annotation) return;
    
    // Immediately position video at segment start time
    if (videoElement.readyState >= 1) { // HAVE_METADATA or better
      videoElement.currentTime = annotation.start_time;
    } else {
      // If metadata not loaded yet, wait for it
      const handleLoadedMetadata = () => {
        videoElement.currentTime = annotation.start_time;
        videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
      };
      videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);
    }
  };

  // Function to handle video play on hover
  const handleVideoPlayback = (event, annotation, isEntering) => {
    const video = event.currentTarget;
    
    if (isEntering) {
      setHoveredSegmentId(annotation.id);
      
      const videoData = videos[annotation.video_id];
      if (!videoData || !videoData.url) return;
      
      // Make sure video is at the start frame
      video.currentTime = annotation.start_time;
      
      // Set up a timeupdate listener to handle looping within segment bounds
      const handleTimeUpdate = () => {
        if (video.currentTime >= annotation.start_time + annotation.duration) {
          video.currentTime = annotation.start_time;
        }
      };
      
      // Remove any existing listener and add a new one
      video.removeEventListener('timeupdate', video._timeUpdateHandler);
      video._timeUpdateHandler = handleTimeUpdate;
      video.addEventListener('timeupdate', video._timeUpdateHandler);
      
      // Start playback
      video.play().catch(e => {
        console.log(`Playback failed for segment ${annotation.id}, showing first frame instead:`, e.message);
      });
    } else {
      // When leaving, pause the video and clean up
      if (hoveredSegmentId === annotation.id) {
        setHoveredSegmentId(null);
      }
      
      // Pause playback and clean up event listeners
      if (video._timeUpdateHandler) {
        video.removeEventListener('timeupdate', video._timeUpdateHandler);
        delete video._timeUpdateHandler;
      }
      
      // Only attempt to pause if the video is actually playing
      if (!video.paused) {
        video.pause();
      }
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto p-4 sm:p-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Segments</h1>
        </div>
        {isCleaningUp && <p className="text-sm text-gray-500 mb-4">Performing initial database cleanup...</p>}
        <div className="grid gap-4">
          {[1, 2, 3, 4, 5, 6].map(i => (
            <Card key={i} className="overflow-hidden">
              <CardContent className="p-0">
                <div className="flex flex-col sm:flex-row">
                  <div className="w-full sm:w-48 h-24 bg-gray-100">
                    <Skeleton className="h-full w-full" />
                  </div>
                  <div className="p-4 flex-grow">
                    <Skeleton className="h-4 w-1/3 mb-2" />
                    <Skeleton className="h-3 w-1/4 mb-4" />
                    <Skeleton className="h-3 w-2/3" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }
  
  return (
    <div className="max-w-7xl mx-auto p-4 sm:p-6">
      {loadingErrors.length > 0 && (
        <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-md">
          <div className="flex items-start gap-2">
            <AlertCircle className="h-5 w-5 text-amber-500 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="font-medium text-amber-800">Some referenced videos are missing</h3>
              <p className="text-sm text-amber-700 mt-1">
                {loadingErrors.length} {loadingErrors.length === 1 ? 'video was' : 'videos were'} not found. Segments will still display but cannot be played.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <h1 className="text-2xl font-bold">Segments</h1>
        <div className="w-full sm:w-auto flex items-center gap-2">
          <div className="relative flex-grow">
            <Search className="h-4 w-4 absolute left-2.5 top-1/2 transform -translate-y-1/2 text-gray-500" />
            <Input
              type="search" placeholder="Search segments..." className="pl-9 w-full"
              value={searchQuery} onChange={handleSearchChange}
            />
          </div>
          <Button 
            variant="outline" size="icon" onClick={() => setExpandedFilters(!expandedFilters)}
            className={expandedFilters ? 'bg-gray-100' : ''}
            aria-expanded={expandedFilters} aria-controls="segment-filters"
          >
            <SlidersHorizontal className="h-4 w-4" />
            <span className="sr-only">Toggle Filters</span>
          </Button>
        </div>
      </div>
      
      <div id="segment-filters" className={`bg-gray-50 p-4 rounded-lg mb-6 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 ${expandedFilters ? '' : 'hidden'}`}>
        <div>
          <label htmlFor="type-filter" className="text-sm font-medium block mb-1">Type</label>
          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger id="type-filter"><SelectValue placeholder="Filter by type" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="SIGN_UNIT">Sign Unit</SelectItem>
              <SelectItem value="BREAK">Break</SelectItem>
              <SelectItem value="FALSE_POSITIVE">False Positive</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label htmlFor="video-filter" className="text-sm font-medium block mb-1">Video</label>
          <Select value={videoFilter} onValueChange={setVideoFilter}>
            <SelectTrigger id="video-filter"><SelectValue placeholder="Filter by video" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Videos</SelectItem>
              {videoOptions.map(video => (
                <SelectItem key={video.id} value={video.id}>{video.title}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <label htmlFor="status-filter" className="text-sm font-medium block mb-1">Status</label>
          <Select value={qualifiedFilter} onValueChange={setQualifiedFilter}>
            <SelectTrigger id="status-filter"><SelectValue placeholder="Filter by status" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Statuses</SelectItem>
              <SelectItem value="qualified">Reviewed</SelectItem>
              <SelectItem value="unqualified">Unreviewed</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label htmlFor="sort-order" className="text-sm font-medium block mb-1">Sort By</label>
          <Select value={sortOrder} onValueChange={setSortOrder}>
            <SelectTrigger id="sort-order"><SelectValue placeholder="Sort by" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="newest">Newest First</SelectItem>
              <SelectItem value="oldest">Oldest First</SelectItem>
              <SelectItem value="duration_asc">Shortest First</SelectItem>
              <SelectItem value="duration_desc">Longest First</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      
      <div className="mb-4 flex items-center justify-between">
        <p className="text-sm text-gray-500">
          {filteredAndSortedAnnotations.length} segments found
        </p>
        {totalPages > 1 && (
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))} disabled={currentPage === 1}>
              <ArrowLeft className="h-4 w-4 mr-1" /> Previous
            </Button>
            <span className="text-sm"> Page {currentPage} of {totalPages} </span>
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))} disabled={currentPage === totalPages}>
              Next <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        )}
      </div>
      
      {paginatedAnnotations.length === 0 ? (
        <div className="bg-white p-12 rounded-xl border-2 border-dashed border-gray-200 text-center">
          <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
            <Film className="h-8 w-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No segments found</h3>
          <p className="text-gray-500 mb-6 max-w-md mx-auto">
            There are no segments matching your current filters. Try adjusting your search or filters, or create new segments in the annotator.
          </p>
          {(searchQuery || typeFilter !== 'all' || videoFilter !== 'all' || qualifiedFilter !== 'all') && (
            <Button variant="outline" onClick={() => {
              setSearchQuery(''); setTypeFilter('all'); setVideoFilter('all'); setQualifiedFilter('all'); setCurrentPage(1);
            }}> Clear All Filters </Button>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {paginatedAnnotations.map(annotation => {
            const video = videos[annotation.video_id];
            const isMissingVideo = !video || !video.url;
            
            return (
              <Card 
                key={annotation.id} 
                className={`overflow-hidden hover:shadow-md transition-shadow ${annotation.qualified ? 'border-l-4 border-l-green-500' : ''} cursor-pointer`}
                onClick={() => handleAnnotationClick(annotation)}
              >
                <CardContent className="p-0">
                  <div className="flex flex-col sm:flex-row">
                    <div className="relative w-full sm:w-48 h-24 bg-gray-800 flex-shrink-0">
                      {isMissingVideo ? (
                        <div className="w-full h-full flex items-center justify-center bg-gray-200">
                          <Film className="h-10 w-10 text-gray-400" />
                          <span className="absolute bottom-1 left-1 text-xs bg-red-500 text-white px-1 py-0.5 rounded-sm">
                            Video Missing
                          </span>
                        </div>
                      ) : (
                        <>
                          <video
                            className="w-full h-full object-cover"
                            src={video.url} 
                            muted
                            playsInline
                            preload="metadata"
                            controls={false}
                            ref={el => el && setupVideoSegmentView(el, annotation)}
                            onMouseEnter={(e) => handleVideoPlayback(e, annotation, true)}
                            onMouseLeave={(e) => handleVideoPlayback(e, annotation, false)}
                          />
                          <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs px-2 py-1 flex items-center justify-between">
                            <span className="truncate max-w-[120px]" title={video.title}>{video.title}</span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {annotation.duration.toFixed(1)}s
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                     <div className="p-4 flex-grow flex flex-col">
                      <div className="flex items-center justify-between">
                        <div className="flex gap-2 flex-wrap">
                          <Badge variant="outline" className={`
                            ${annotation.segment_type === 'SIGN_UNIT' ? 'bg-blue-100 text-blue-800 border-blue-200' : ''}
                            ${annotation.segment_type === 'BREAK' ? 'bg-amber-100 text-amber-800 border-amber-200' : ''}
                            ${annotation.segment_type === 'TRANSITION' ? 'bg-purple-100 text-purple-800 border-purple-200' : ''}
                            ${annotation.segment_type === 'FALSE_POSITIVE' ? 'bg-gray-100 text-gray-800 border-gray-200' : ''}
                            ${annotation.segment_type === 'SEGMENT' || !annotation.segment_type ? 'bg-indigo-100 text-indigo-800 border-indigo-200' : ''}
                          `}>
                            <span className="flex items-center gap-1">
                              {getSegmentIcon(annotation.segment_type || 'SEGMENT')}
                              {annotation.segment_type === 'SIGN_UNIT' ? 'Sign Unit' : ''}
                              {annotation.segment_type === 'BREAK' ? 'Break' : ''}
                              {annotation.segment_type === 'TRANSITION' ? 'Transition' : ''}
                              {annotation.segment_type === 'FALSE_POSITIVE' ? 'False Positive' : ''}
                              {annotation.segment_type === 'SEGMENT' || !annotation.segment_type ? 'Segment' : ''}
                            </span>
                          </Badge>
                          <Badge variant="outline" className="bg-gray-100">
                            {formatTime(annotation.start_time)} → {formatTime(annotation.start_time + annotation.duration)}
                          </Badge>
                          {annotation.qualified && (
                            <Badge variant="outline" className="bg-green-100 text-green-800 border-green-300 flex items-center gap-1">
                              <Check className="h-3 w-3" />
                              Validated
                            </Badge>
                          )}
                        </div>
                        
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button 
                                  variant="ghost" size="icon" className="h-8 w-8 p-0 shrink-0"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                    <MoreVertical className="h-4 w-4" />
                                    <span className="sr-only">Segment Options</span>
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                                <DropdownMenuItem 
                                    onClick={() => handleShareSegment(annotation)}
                                    className="flex items-center gap-2 cursor-pointer" disabled={isMissingVideo}
                                > <Share2 className="h-4 w-4" /> Share Link </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem 
                                    onClick={() => confirmDeleteAnnotation(annotation)} 
                                    className="flex items-center gap-2 text-red-600 hover:!text-red-700 hover:!bg-red-50 cursor-pointer"
                                > <Trash2 className="h-4 w-4" /> Delete Segment </DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      
                      <p className="mt-2 text-sm text-gray-700 line-clamp-2" title={annotation.label || annotation.description || 'No description provided'}>
                        {annotation.label || annotation.description || 'No description provided'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
      
      {totalPages > 1 && (
        <div className="mt-6 flex justify-center">
          <div className="flex items-center gap-1">
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(1)} disabled={currentPage === 1}>First</Button>
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))} disabled={currentPage === 1}><ArrowLeft className="h-4 w-4" /></Button>
            <span className="px-4 text-sm">{currentPage} of {totalPages}</span>
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))} disabled={currentPage === totalPages}><ArrowRight className="h-4 w-4" /></Button>
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(totalPages)} disabled={currentPage === totalPages}>Last</Button>
          </div>
        </div>
      )}
      
      <AnnotationDetailDialog
        isOpen={isDetailDialogOpen}
        onClose={() => { setIsDetailDialogOpen(false); setSelectedAnnotation(null); }}
        annotation={selectedAnnotation}
        onSave={handleAnnotationUpdate}
        videoDuration={selectedAnnotation && videos[selectedAnnotation.video_id] ? (videos[selectedAnnotation.video_id].duration || 0) : 0}
        videoUrl={selectedAnnotation && videos[selectedAnnotation.video_id] && !videos[selectedAnnotation.video_id].missing ? videos[selectedAnnotation.video_id].url : ''}
      />

      <AlertDialog open={!!annotationToDelete} onOpenChange={() => setAnnotationToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Segment?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove the segment: "{annotationToDelete?.label || annotationToDelete?.description || 'Selected Segment'}". This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction className="bg-red-600 hover:bg-red-700" onClick={executeDeleteAnnotation}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
