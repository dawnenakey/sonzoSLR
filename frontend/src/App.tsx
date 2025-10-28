import React, { useState, useEffect } from 'react';
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Progress } from "./components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./components/ui/dialog";
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./components/ui/tooltip";
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";
import { 
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";
import { 
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "./components/ui/tabs";
import { 
  Camera, Video, Play, Pause, RotateCcw, Download, Upload, 
  CheckCircle, AlertCircle, Loader2, Eye, EyeOff, Settings,
  Zap, Brain, Search, FileVideo, Clock, BarChart3, Home, User, Settings as SettingsIcon, Database
} from 'lucide-react';
import { useToast } from "./components/ui/use-toast";
import AdvancedSignSpotting from "./components/AdvancedSignSpotting";
import CameraTest from "./pages/CameraTest";
import { awsAPI } from './api/awsClient';

interface Annotation {
  id: string;
  startTime: number;
  endTime: number;
  label: string;
  confidence?: number;
  notes?: string;
}

interface VideoData {
  file: File;
  url: string;
  metadata: {
    name: string;
    size: number;
    type: string;
    source: string;
    timestamp: string;
  };
}

function App() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [currentSession, setCurrentSession] = useState<any | null>(null);
  const [currentVideo, setCurrentVideo] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [showAdvancedFeatures, setShowAdvancedFeatures] = useState(true);
  const [processedVideoData, setProcessedVideoData] = useState<VideoData | null>(null);
  const [currentView, setCurrentView] = useState('home'); // home, camera, analysis

  const { toast } = useToast();

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
    checkConnectionHealth();
  }, []);

  const checkConnectionHealth = async () => {
    try {
      // Check API connectivity
      const response = await fetch('/api/health', { 
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('degraded');
      }
    } catch (err) {
      setConnectionStatus('disconnected');
      console.warn('Connection check failed:', err);
    }
  };

  const loadSessions = async () => {
    try {
      // Note: awsAPI.sessions.list() might not exist, so we'll handle this gracefully
      // setSessions(await awsAPI.sessions.list());
      console.log('Sessions loading not implemented yet');
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const handleFileSelect = async (file: File) => {
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
    setCurrentVideo(null);
  };

  const handleAcceptAndUpload = async () => {
    if (!selectedFile) {
      toast({
        variant: "destructive",
        title: "No file selected",
        description: "Please select a video file to upload"
      });
      return;
    }
    setIsUploading(true);
    try {
      // Create a new session if none exists
      let session = currentSession;
      if (!session) {
        session = await awsAPI.sessions.create({ 
          name: selectedFile.name, 
          description: "Video uploaded from annotation tool" 
        });
        setCurrentSession(session);
        await loadSessions();
      }
      // Upload video to the session
      const video = await awsAPI.sessions.uploadVideo(session.id, selectedFile);
      console.log('Uploaded video object:', video);
      if (!video.id) {
        toast({
          variant: "destructive",
          title: "Upload Error",
          description: "Upload succeeded but video.id is missing!"
        });
      }
      setCurrentVideo(video);
      toast({
        title: "Upload Successful",
        description: "File uploaded successfully! You can now start annotating."
      });
    } catch (error) {
      console.error("Upload failed:", error);
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: `Upload failed: ${error}`
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleCreateAnnotation = async (startTime: number, endTime: number, label: string, notes?: string) => {
    if (!currentVideo) return;
    try {
      const annotation = await awsAPI.annotations.create({
        videoId: currentVideo.id,
        startTime,
        endTime,
        label,
        confidence: 1.0,
        notes,
      });
      setAnnotations((prev: Annotation[]) => [...prev, annotation]);
      toast({
        title: "Annotation Created",
        description: "New annotation has been created successfully"
      });
    } catch (error) {
      console.error('Failed to create annotation:', error);
      toast({
        variant: "destructive",
        title: "Annotation Failed",
        description: "Failed to create annotation"
      });
    }
  };

  const handleUpdateAnnotation = async (annotationId: string, updates: any) => {
    if (!currentVideo) return;
    try {
      const updatedAnnotation = await awsAPI.annotations.update(annotationId, updates);
      setAnnotations((prev: Annotation[]) => prev.map((a: Annotation) => a.id === annotationId ? updatedAnnotation : a));
      toast({
        title: "Annotation Updated",
        description: "Annotation has been updated successfully"
      });
    } catch (error) {
      console.error('Failed to update annotation:', error);
      toast({
        variant: "destructive",
        title: "Update Failed",
        description: "Failed to update annotation"
      });
    }
  };

  const handleDeleteAnnotation = async (annotationId: string) => {
    if (!currentVideo) return;
    try {
      await awsAPI.annotations.delete(annotationId);
      setAnnotations((prev: Annotation[]) => prev.filter((a: Annotation) => a.id !== annotationId));
      toast({
        title: "Annotation Deleted",
        description: "Annotation has been deleted successfully"
      });
    } catch (error) {
      console.error('Failed to delete annotation:', error);
      toast({
        variant: "destructive",
        title: "Delete Failed",
        description: "Failed to delete annotation"
      });
    }
  };

  const handleVideoUploaded = async (file: File) => {
    try {
      setIsUploading(true);
      
      // Create video data object for enhanced viewer
      const videoData: VideoData = {
        file: file,
        url: URL.createObjectURL(file),
        metadata: {
          name: file.name,
          size: file.size,
          type: file.type,
          source: 'live-camera',
          timestamp: new Date().toISOString()
        }
      };
      
      setProcessedVideoData(videoData);
      
      toast({
        title: "Video Processed",
        description: "Video has been processed and is ready for analysis"
      });
      
    } catch (error) {
      console.error('Error processing video:', error);
      toast({
        variant: "destructive",
        title: "Processing Failed",
        description: "Could not process the uploaded video"
      });
    } finally {
      setIsUploading(false);
    }
  };

  // Load annotations when video changes
  useEffect(() => {
    if (currentVideo) {
      awsAPI.annotations.getByVideo(currentVideo.id).then(setAnnotations);
    }
  }, [currentVideo]);

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'disconnected': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'degraded': return 'Slow Connection';
      case 'disconnected': return 'Disconnected';
      default: return 'Unknown';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <div className="flex items-center gap-4 mb-4">
            {/* SpokHand Logo */}
            <div className="flex items-center gap-3">
              <img src="/spokhand-icon.svg" alt="SpokHand Logo" className="w-16 h-16" />
              <div>
                <h1 className="text-4xl font-bold text-gray-900 mb-2">SPOKHAND SIGNCUT</h1>
                <p className="text-gray-600">Advanced Sign Language Recognition System</p>
              </div>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Navigation Sidebar */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Navigation</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button
                  variant={currentView === 'home' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('home')}
                >
                  <Home className="h-4 w-4 mr-2" />
                  Home
                </Button>
                <Button
                  variant={currentView === 'camera' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('camera')}
                >
                  <Camera className="h-4 w-4 mr-2" />
                  Camera
                </Button>
                <Button
                  variant={currentView === 'camera-test' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('camera-test')}
                >
                  <Video className="h-4 w-4 mr-2" />
                  Camera Test
                </Button>
                <Button
                  variant={currentView === 'camera-settings' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('camera-settings')}
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Camera Settings
                </Button>
                <Button
                  variant={currentView === 'analysis' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('analysis')}
                >
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Analysis
                </Button>
                <Button
                  variant={currentView === 'database' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('database')}
                >
                  <Database className="h-4 w-4 mr-2" />
                  ASL-LEX Database
                </Button>
                <Button
                  variant={currentView === 'troubleshoot' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('troubleshoot')}
                >
                  <AlertCircle className="h-4 w-4 mr-2" />
                  Troubleshoot
                </Button>
                <Button
                  variant={currentView === 'settings' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setCurrentView('settings')}
                >
                  <SettingsIcon className="h-4 w-4 mr-2" />
                  Settings
                </Button>
              </CardContent>
            </Card>

            {/* Connection Status */}
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-500' :
                    connectionStatus === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  <span className="text-sm capitalize">{connectionStatus}</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2">
            {currentView === 'home' && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Welcome to SpokHand SLR</CardTitle>
                    <CardDescription>
                      Advanced Sign Language Recognition and Data Management
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex justify-between items-center mb-4">
                      <div className="flex gap-2">
                        <Button
                          variant={showAdvancedFeatures ? 'default' : 'outline'}
                          onClick={() => setShowAdvancedFeatures(!showAdvancedFeatures)}
                          className="flex items-center gap-2"
                        >
                          {showAdvancedFeatures ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                          {showAdvancedFeatures ? 'Hide Advanced' : 'Show Advanced'}
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => setCurrentView('database')}
                          className="flex items-center gap-2"
                        >
                          <Database className="h-4 w-4" />
                          Show Database
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => setCurrentView('analysis')}
                          className="flex items-center gap-2"
                        >
                          <BarChart3 className="h-4 w-4" />
                          Show ASL-LEX
                        </Button>
                      </div>
                    </div>
                    
                    {showAdvancedFeatures && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div className="p-4 border rounded-lg bg-blue-50">
                          <h3 className="font-semibold mb-2 flex items-center gap-2">
                            <Brain className="h-4 w-4 text-blue-600" />
                            AI Advanced Features
                          </h3>
                          <p className="text-sm text-gray-600">
                            Advanced AI-powered sign language recognition with I3D, ResNeXt-101, and LLM disambiguation
                          </p>
                        </div>
                        <div className="p-4 border rounded-lg bg-green-50">
                          <h3 className="font-semibold mb-2 flex items-center gap-2">
                            <Zap className="h-4 w-4 text-green-600" />
                            Enhanced Analysis
                          </h3>
                          <p className="text-sm text-gray-600">
                            Real-time video analysis, sign spotting, and context-aware disambiguation
                          </p>
                        </div>
                      </div>
                    )}
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 border rounded-lg">
                        <h3 className="font-semibold mb-2">Real-time Recognition</h3>
                        <p className="text-sm text-gray-600">
                          Use your camera to recognize ASL signs in real-time
                        </p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <h3 className="font-semibold mb-2">Data Management</h3>
                        <p className="text-sm text-gray-600">
                          Manage your ASL-LEX database with advanced tools
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {currentView === 'camera' && (
              <Card>
                <CardHeader>
                  <CardTitle>Camera Recognition</CardTitle>
                  <CardDescription>
                    Real-time ASL sign recognition using your camera
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-gray-600">Camera functionality coming soon...</p>
                  <Button 
                    onClick={() => setCurrentView('camera-test')}
                    className="flex items-center gap-2"
                  >
                    <Video className="h-4 w-4" />
                    Test Your Camera
                  </Button>
                </CardContent>
              </Card>
            )}

            {currentView === 'camera-test' && (
              <CameraTest />
            )}

            {currentView === 'camera-settings' && (
              <Card>
                <CardHeader>
                  <CardTitle>Camera Settings</CardTitle>
                  <CardDescription>
                    Configure camera parameters and preferences
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold mb-2">Camera Configuration</h3>
                      <p className="text-gray-600">Adjust resolution, frame rate, and other camera settings</p>
                    </div>
                    <div>
                      <h3 className="font-semibold mb-2">Device Selection</h3>
                      <p className="text-gray-600">Choose between BRIO, OAK-D, or other connected cameras</p>
                    </div>
                    <div>
                      <h3 className="font-semibold mb-2">Permissions</h3>
                      <p className="text-gray-600">Manage camera access permissions and privacy settings</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {currentView === 'analysis' && (
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Dashboard</CardTitle>
                  <CardDescription>
                    View analytics and performance metrics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">Analysis dashboard coming soon...</p>
                </CardContent>
              </Card>
            )}

            {currentView === 'database' && (
              <div className="text-center py-8">
                <p className="text-gray-500">Database functionality available in other components</p>
              </div>
            )}

            {currentView === 'troubleshoot' && (
              <Card>
                <CardHeader>
                  <CardTitle>Camera Troubleshooting</CardTitle>
                  <CardDescription>
                    Common camera issues and solutions
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-4">
                    <div className="p-4 border rounded-lg">
                      <h3 className="font-semibold mb-2">BRIO Camera Issues</h3>
                      <p className="text-sm text-gray-600 mb-2">Common problems and solutions for Logitech BRIO cameras</p>
                      <ul className="text-sm text-gray-600 space-y-1">
                        <li>• Check Mac privacy settings in System Preferences</li>
                        <li>• Ensure camera is not in use by other applications</li>
                        <li>• Try different USB ports</li>
                        <li>• Update camera drivers</li>
                      </ul>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h3 className="font-semibold mb-2">OAK-D Camera Issues</h3>
                      <p className="text-sm text-gray-600 mb-2">Troubleshooting for OAK-D depth cameras</p>
                      <ul className="text-sm text-gray-600 space-y-1">
                        <li>• Verify USB-C connection</li>
                        <li>• Check depthai library installation</li>
                        <li>• Ensure proper power supply</li>
                        <li>• Test with depthai examples</li>
                      </ul>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h3 className="font-semibold mb-2">General Camera Problems</h3>
                      <p className="text-sm text-gray-600 mb-2">Universal camera troubleshooting</p>
                      <ul className="text-sm text-gray-600 space-y-1">
                        <li>• Refresh browser permissions</li>
                        <li>• Clear browser cache and cookies</li>
                        <li>• Try different browsers</li>
                        <li>• Check system camera permissions</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {currentView === 'settings' && (
              <Card>
                <CardHeader>
                  <CardTitle>Application Settings</CardTitle>
                  <CardDescription>
                    Configure application preferences and system settings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold mb-2">General Preferences</h3>
                      <p className="text-gray-600">Application behavior and interface settings</p>
                    </div>
                    <div>
                      <h3 className="font-semibold mb-2">API Configuration</h3>
                      <p className="text-gray-600">AWS and external service settings</p>
                    </div>
                    <div>
                      <h3 className="font-semibold mb-2">User Management</h3>
                      <p className="text-gray-600">Account settings and permissions</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
