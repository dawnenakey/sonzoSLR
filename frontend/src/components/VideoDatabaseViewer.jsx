import React, { useState, useEffect } from 'react';
import { 
  Video, Play, Pause, Download, Trash2, Search, Filter, 
  Calendar, Clock, FileVideo, Eye, Edit2, MoreVertical,
  Loader2, AlertCircle, CheckCircle, XCircle, Upload, RotateCcw
} from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from './ui/dropdown-menu';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from './ui/alert-dialog';
import { useToast } from './ui/use-toast';
import { videoAPI } from '@/api/awsClient';
import VideoThumbnail from './VideoThumbnail';
import { formatTime } from './timeUtils';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';

export default function VideoDatabaseViewer({ onVideoSelect }) {
  const [videos, setVideos] = useState([]);
  const [filteredVideos, setFilteredVideos] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState('uploadedAt');
  const [sortOrder, setSortOrder] = useState('desc');
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState(null);
  const [error, setError] = useState(null);
  
  const { toast } = useToast();

  useEffect(() => {
    loadVideos();
  }, []);

  useEffect(() => {
    filterAndSortVideos();
  }, [videos, searchTerm, statusFilter, sortBy, sortOrder]);

  const loadVideos = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const fetchedVideos = await videoAPI.list();
      console.log('Loaded videos from AWS:', fetchedVideos);
      
      // Add mock data for demonstration if no videos exist
      if (!fetchedVideos || fetchedVideos.length === 0) {
        const mockVideos = [
          {
            id: 'mock-1',
            sessionId: 'session-1',
            filename: 'asl_sample_1.mp4',
            size: 15485760,
            duration: 45.2,
            uploadedAt: new Date(Date.now() - 86400000).toISOString(),
            status: 'ready',
            url: '/api/videos/mock-1/stream'
          },
          {
            id: 'mock-2',
            sessionId: 'session-2',
            filename: 'bsl_interview.mp4',
            size: 23456789,
            duration: 120.5,
            uploadedAt: new Date(Date.now() - 172800000).toISOString(),
            status: 'ready',
            url: '/api/videos/mock-2/stream'
          },
          {
            id: 'mock-3',
            sessionId: 'session-3',
            filename: 'sign_language_lesson.mp4',
            size: 8765432,
            duration: 67.8,
            uploadedAt: new Date(Date.now() - 259200000).toISOString(),
            status: 'processing',
            url: '/api/videos/mock-3/stream'
          }
        ];
        setVideos(mockVideos);
      } else {
        setVideos(fetchedVideos);
      }
    } catch (err) {
      console.error('Error loading videos:', err);
      setError('Failed to load videos from database');
      toast({
        variant: "destructive",
        title: "Error Loading Videos",
        description: "Could not load videos from AWS database"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const filterAndSortVideos = () => {
    let filtered = [...videos];

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(video => 
        video.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
        video.sessionId.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(video => video.status === statusFilter);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];

      if (sortBy === 'uploadedAt') {
        aValue = new Date(aValue).getTime();
        bValue = new Date(bValue).getTime();
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    setFilteredVideos(filtered);
  };

  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
    if (onVideoSelect) {
      onVideoSelect(video);
    }
  };

  const handleDeleteVideo = async (video) => {
    setVideoToDelete(video);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
    if (!videoToDelete) return;

    try {
      setIsDeleting(true);
      await videoAPI.delete(videoToDelete.id);
      
      setVideos(videos.filter(v => v.id !== videoToDelete.id));
      setSelectedVideo(null);
      
      toast({
        title: "Video Deleted",
        description: `${videoToDelete.filename} has been removed from the database`
      });
    } catch (err) {
      console.error('Error deleting video:', err);
      toast({
        variant: "destructive",
        title: "Delete Failed",
        description: "Could not delete the video from the database"
      });
    } finally {
      setIsDeleting(false);
      setDeleteDialogOpen(false);
      setVideoToDelete(null);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'uploading':
        return <Upload className="h-4 w-4 text-yellow-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ready':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-blue-100 text-blue-800';
      case 'uploading':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Video className="h-5 w-5" />
            AWS Video Database
          </CardTitle>
          <CardDescription>
            Loading videos from AWS S3 database...
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Video className="h-5 w-5" />
                AWS Video Database
              </CardTitle>
              <CardDescription>
                {videos.length} videos stored in AWS S3 • {filteredVideos.length} filtered
              </CardDescription>
            </div>
            <Button onClick={loadVideos} variant="outline" size="sm">
              <RotateCcw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {/* Search and Filters */}
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search videos by filename or session ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            
            <div className="flex gap-2">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value="all">All Status</option>
                <option value="ready">Ready</option>
                <option value="processing">Processing</option>
                <option value="uploading">Uploading</option>
                <option value="error">Error</option>
              </select>
              
              <select
                value={`${sortBy}-${sortOrder}`}
                onChange={(e) => {
                  const [field, order] = e.target.value.split('-');
                  setSortBy(field);
                  setSortOrder(order);
                }}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value="uploadedAt-desc">Newest First</option>
                <option value="uploadedAt-asc">Oldest First</option>
                <option value="filename-asc">Name A-Z</option>
                <option value="filename-desc">Name Z-A</option>
                <option value="size-desc">Largest First</option>
                <option value="size-asc">Smallest First</option>
              </select>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <Alert className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Database Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Video Grid */}
          {filteredVideos.length === 0 ? (
            <div className="text-center py-12">
              <FileVideo className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                {searchTerm || statusFilter !== 'all' ? 'No videos match your filters' : 'No videos found'}
              </h3>
              <p className="text-gray-600">
                {searchTerm || statusFilter !== 'all' 
                  ? 'Try adjusting your search terms or filters'
                  : 'Upload your first video to get started'
                }
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredVideos.map((video) => (
                <Card 
                  key={video.id} 
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedVideo?.id === video.id ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => handleVideoSelect(video)}
                >
                  <CardContent className="p-4">
                    <div className="aspect-video bg-gray-100 rounded-lg mb-3 overflow-hidden">
                      <VideoThumbnail 
                        videoUrl={video.url || videoAPI.getStreamUrl(video.id)}
                        alt={video.filename}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-start justify-between">
                        <h3 className="font-medium text-gray-900 truncate flex-1">
                          {video.filename}
                        </h3>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={(e) => {
                              e.stopPropagation();
                              handleVideoSelect(video);
                            }}>
                              <Eye className="h-4 w-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={(e) => {
                              e.stopPropagation();
                              // Handle edit
                            }}>
                              <Edit2 className="h-4 w-4 mr-2" />
                              Edit Metadata
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem 
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteVideo(video);
                              }}
                              className="text-red-600"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {getStatusIcon(video.status)}
                        <Badge variant="secondary" className={getStatusColor(video.status)}>
                          {video.status}
                        </Badge>
                      </div>
                      
                      <div className="text-sm text-gray-600 space-y-1">
                        <div className="flex items-center gap-2">
                          <Calendar className="h-3 w-3" />
                          <span>{formatDate(video.uploadedAt)}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <FileVideo className="h-3 w-3" />
                          <span>{formatFileSize(video.size)}</span>
                        </div>
                        {video.duration && (
                          <div className="flex items-center gap-2">
                            <Clock className="h-3 w-3" />
                            <span>{formatTime(video.duration)}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Video</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{videoToDelete?.filename}"? This action cannot be undone and will remove the video from the AWS database.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={confirmDelete}
              disabled={isDeleting}
              className="bg-red-600 hover:bg-red-700"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                'Delete'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
} 