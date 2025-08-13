import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { createPageUrl } from '@/utils';
import { Film, Plus, ArrowRight, Play, Calendar, Clock, Trash2, MoreVertical, Edit2, AlertCircle, UserCircle, Download, Share2, Camera, Upload, EyeOff, Eye, Loader2, Database, BookOpen } from 'lucide-react'; // Added Database and BookOpen
import { Button } from '@/components/ui/button';
import { AnnotationEntity } from '@/api/entities';
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
import EnhancedVideoViewer from '../components/EnhancedVideoViewer';
import CameraSelector from '../components/CameraSelector';
import VideoDatabaseViewer from '../components/VideoDatabaseViewer';
import AdvancedSignSpotting from '../components/AdvancedSignSpotting';
import ASLLexDataManager from '../components/ASLLexDataManager';
import { videoAPI } from '@/api/awsClient';
import VideoThumbnail from '../components/VideoThumbnail';

export default function Home() {
  const [videos, setVideos] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showAdvancedFeatures, setShowAdvancedFeatures] = useState(true);
  const [processedVideoData, setProcessedVideoData] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [cameraSettings, setCameraSettings] = useState({});
  const [showDatabase, setShowDatabase] = useState(false);
  const [showASLLex, setShowASLLex] = useState(true);
  
  const { toast } = useToast();

  useEffect(() => {
    loadVideos();
  }, []);

  const loadVideos = async () => {
    try {
      setIsLoading(true);
      const fetchedVideos = await videoAPI.list();
      setVideos(fetchedVideos || []);
    } catch (error) {
      console.error('Error loading videos:', error);
      toast({
        variant: "destructive",
        title: "Error Loading Videos",
        description: "Could not load videos from database"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleVideoUploaded = (videoData) => {
    setProcessedVideoData(videoData);
    
    // Add the uploaded video to the videos list
    const newVideo = {
      id: `uploaded-${Date.now()}`,
      filename: videoData.file?.name || 'Uploaded Video',
      size: videoData.file?.size || 0,
      status: 'ready',
      uploadedAt: new Date().toISOString(),
      url: videoData.url
    };
    
    setVideos(prev => [...prev, newVideo]);
    
    toast({
      title: "Video Processed",
      description: "Video has been processed and is ready for analysis"
    });
  };

  const handleVideoSelect = (video) => {
    // Navigate to annotator with the selected video
    window.location.href = createPageUrl('annotator', { id: video.id });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Sign Language Annotation Tool</h1>
              <p className="text-lg text-gray-600">
                Advanced AI-powered sign language recognition and annotation
              </p>
            </div>
            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => setShowAdvancedFeatures(!showAdvancedFeatures)}
                className="flex items-center gap-2"
              >
                {showAdvancedFeatures ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                {showAdvancedFeatures ? 'Hide' : 'Show'} Advanced
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowDatabase(!showDatabase)}
                className="flex items-center gap-2"
              >
                <Database className="h-4 w-4" />
                {showDatabase ? 'Hide' : 'Show'} Database
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowASLLex(!showASLLex)}
                className="flex items-center gap-2"
              >
                <BookOpen className="h-4 w-4" />
                {showASLLex ? 'Hide' : 'Show'} ASL-LEX
              </Button>
              <Link to={createPageUrl("ASLLex")}>
                <Button
                  variant="default"
                  className="flex items-center gap-2"
                >
                  <BookOpen className="h-4 w-4" />
                  Full ASL-LEX Manager
                </Button>
              </Link>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="flex items-center gap-3">
                <div className="bg-blue-100 p-2 rounded-lg">
                  <Film className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Total Videos</p>
                  <p className="text-2xl font-bold text-gray-900">{videos.length}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="flex items-center gap-3">
                <div className="bg-green-100 p-2 rounded-lg">
                  <Camera className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Live Camera</p>
                  <p className="text-2xl font-bold text-gray-900">Ready</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="flex items-center gap-3">
                <div className="bg-purple-100 p-2 rounded-lg">
                  <Eye className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">AI Analysis</p>
                  <p className="text-2xl font-bold text-gray-900">Active</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="flex items-center gap-3">
                <div className="bg-amber-100 p-2 rounded-lg">
                  <Database className="h-5 w-5 text-amber-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">AWS Storage</p>
                  <p className="text-2xl font-bold text-gray-900">Connected</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Live Camera Section */}
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
              <LiveCameraAnnotator 
                onVideoUploaded={handleVideoUploaded}
              />
            </div>
          </div>
        </section>

        {/* Enhanced Video Viewer */}
        {processedVideoData && (
          <section className="mb-6">
            <EnhancedVideoViewer
              videoData={processedVideoData}
              showAdvancedFeatures={showAdvancedFeatures}
              onVideoProcessed={(data) => {
                console.log('Video processed:', data);
              }}
            />
          </section>
        )}
        
        {/* AWS Video Database Section */}
        {showDatabase && (
          <section className="mb-6">
            <VideoDatabaseViewer onVideoSelect={handleVideoSelect} />
          </section>
        )}

        {/* ASL-LEX Data Manager Section */}
        {showASLLex && (
          <section className="mb-6">
            <ASLLexDataManager />
          </section>
        )}

        {/* Advanced Sign Spotting Section - Only show when advanced features are enabled and there's video data */}
        {showAdvancedFeatures && processedVideoData && (
          <section className="mb-6">
            <AdvancedSignSpotting
              videoRef={null}
              analysisResults={null}
              isAnalyzing={false}
              onAnalyze={() => {
                console.log('Advanced analysis triggered');
              }}
            />
          </section>
        )}
        
        {/* Video Library Section */}
        <section className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Video Library</h2>
            <ImportVideoDialog onFileSelect={(file) => {
              // Handle file selection
              console.log('File selected:', file);
            }} />
          </div>
          
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="bg-white rounded-lg shadow-sm p-4">
                  <Skeleton className="h-32 w-full mb-3" />
                  <Skeleton className="h-4 w-3/4 mb-2" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              ))}
            </div>
          ) : videos.length === 0 ? (
            <div className="text-center py-12 bg-white rounded-lg shadow-sm">
              <Film className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No videos yet</h3>
              <p className="text-gray-600 mb-4">Upload your first video to get started</p>
              <ImportVideoDialog onFileSelect={(file) => {
                // Handle file selection
                console.log('File selected:', file);
              }} />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {videos.map((video) => (
                <div key={video.id} className="bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow">
                  <div className="aspect-video bg-gray-100 relative">
                    <VideoThumbnail 
                      videoUrl={video.url || videoAPI.getStreamUrl(video.id)}
                      alt={video.filename}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute top-2 right-2">
                      <Badge 
                        variant="secondary" 
                        className={
                          video.status === 'ready' ? 'bg-green-100 text-green-800' :
                          video.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                          video.status === 'uploading' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }
                      >
                        {video.status}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="p-4">
                    <div className="flex items-start justify-between mb-2">
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
                          <DropdownMenuItem onClick={() => handleVideoSelect(video)}>
                            <Play className="h-4 w-4 mr-2" />
                            Annotate
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Edit2 className="h-4 w-4 mr-2" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem className="text-red-600">
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    
                    <div className="text-sm text-gray-600 space-y-1">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-3 w-3" />
                        <span>{new Date(video.uploadedAt).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="h-3 w-3" />
                        <span>{video.duration ? formatTime(video.duration) : 'Unknown duration'}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <UserCircle className="h-3 w-3" />
                        <span>Session: {video.sessionId}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
