import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { 
  Camera, Video, Play, Pause, RotateCcw, Download, Upload, 
  CheckCircle, AlertCircle, Loader2, Eye, EyeOff, Settings,
  Zap, Brain, Search, FileVideo, Clock, BarChart3
} from 'lucide-react';
import { useToast } from './ui/use-toast';
import VideoPlayer from './VideoPlayer';
import AdvancedSignSpotting from './AdvancedSignSpotting';

export default function EnhancedVideoViewer({ 
  videoData, 
  onVideoProcessed, 
  showAdvancedFeatures = true 
}) {
  const [isLoading, setIsLoading] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('idle');
  const [processedData, setProcessedData] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [showAdvancedUI, setShowAdvancedUI] = useState(showAdvancedFeatures);
  const [videoUrl, setVideoUrl] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  
  const videoRef = useRef(null);
  const { toast } = useToast();

  // Connection health check
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('/api/health', { 
          method: 'GET',
          signal: AbortSignal.timeout(5000) // 5 second timeout
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

    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    checkConnection(); // Initial check

    return () => clearInterval(interval);
  }, []);

  // Process video data when it changes
  useEffect(() => {
    if (videoData) {
      processVideoData(videoData);
    }
  }, [videoData]);

  const processVideoData = async (data) => {
    setIsLoading(true);
    setError(null);
    setProcessingStatus('processing');

    try {
      // Handle different video data formats
      let videoUrl = null;
      let metadata = {};

      if (typeof data === 'string') {
        // Direct URL
        videoUrl = data;
        metadata = { source: 'url' };
      } else if (data && data.url) {
        // Object with URL
        videoUrl = data.url;
        metadata = data.metadata || data;
      } else if (data && data.file) {
        // File object from upload
        videoUrl = URL.createObjectURL(data.file);
        metadata = {
          name: data.file.name,
          size: data.file.size,
          type: data.file.type,
          source: 'file'
        };
      } else if (data && data.metadata) {
        // Object with metadata
        videoUrl = data.url || data.metadata.url;
        metadata = data.metadata;
      } else {
        console.warn('Unknown video data format:', data);
        // Don't throw error, just show warning
        setProcessingStatus('completed');
        setIsLoading(false);
        return;
      }

      if (!videoUrl) {
        console.warn('No video URL found in data:', data);
        setProcessingStatus('completed');
        setIsLoading(false);
        return;
      }

      setVideoUrl(videoUrl);
      setProcessedData({ url: videoUrl, metadata });

      // Auto-analyze if advanced features are enabled
      if (showAdvancedUI) {
        await analyzeVideo(videoUrl, metadata);
      }

      setProcessingStatus('completed');
      if (onVideoProcessed) {
        onVideoProcessed({ url: videoUrl, metadata });
      }

    } catch (err) {
      console.error('Error processing video:', err);
      setError(err.message);
      setProcessingStatus('error');
      toast({
        variant: "destructive",
        title: "Processing Failed",
        description: err.message
      });
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeVideo = async (url, metadata) => {
    setIsAnalyzing(true);
    setAnalysisResults(null);

    try {
      // Simulate advanced sign spotting analysis
      const mockResults = {
        signSpots: [
          { start: 2.5, end: 4.2, confidence: 0.89, gloss: 'HELLO' },
          { start: 6.1, end: 8.3, confidence: 0.92, gloss: 'HOW' },
          { start: 9.5, end: 11.8, confidence: 0.85, gloss: 'ARE' },
          { start: 13.2, end: 15.7, confidence: 0.91, gloss: 'YOU' }
        ],
        handShapes: [
          { time: 3.1, shape: 'B', confidence: 0.87 },
          { time: 7.2, shape: 'A', confidence: 0.93 },
          { time: 10.8, shape: 'C', confidence: 0.79 }
        ],
        disambiguation: {
          originalSequence: ['HELLO', 'HOW', 'ARE', 'YOU'],
          disambiguatedSequence: ['HELLO', 'HOW', 'ARE', 'YOU'],
          confidence: 0.94
        }
      };

      // Simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setAnalysisResults(mockResults);
      toast({
        title: "Analysis Complete",
        description: "Advanced sign spotting analysis finished successfully"
      });

    } catch (err) {
      console.error('Analysis failed:', err);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: "Could not complete advanced analysis"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

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

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="flex items-center justify-center space-x-4">
            <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
            <div>
              <h3 className="text-lg font-semibold">Processing Video</h3>
              <p className="text-sm text-gray-600">
                {processingStatus === 'processing' ? 'Analyzing video data...' : 
                 processingStatus === 'error' ? 'Processing failed' : 'Loading...'}
              </p>
            </div>
          </div>
          {processingStatus === 'processing' && (
            <Progress value={75} className="mt-4" />
          )}
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`}></div>
          <span className="text-sm text-gray-600">{getConnectionStatusText()}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAdvancedUI(!showAdvancedUI)}
            className="flex items-center space-x-2"
          >
            {showAdvancedUI ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            <span>{showAdvancedUI ? 'Hide' : 'Show'} Advanced</span>
          </Button>
        </div>
      </div>

      {/* Video Player */}
      {videoUrl && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center space-x-2">
                  <Video className="h-5 w-5" />
                  <span>Video Analysis</span>
                  {processedData?.metadata?.name && (
                    <Badge variant="secondary">{processedData.metadata.name}</Badge>
                  )}
                </CardTitle>
                <CardDescription>
                  Advanced sign language recognition and analysis
                </CardDescription>
              </div>
              {isAnalyzing && (
                <div className="flex items-center space-x-2 text-blue-600">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm">Analyzing...</span>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <VideoPlayer
              videoUrl={videoUrl}
              videoRef={videoRef}
              showMainControls={true}
              containerClassName="relative w-full bg-black rounded-lg overflow-hidden"
            />
          </CardContent>
        </Card>
      )}

      {/* Advanced Sign Spotting */}
      {showAdvancedUI && (
        <AdvancedSignSpotting
          videoRef={videoRef}
          analysisResults={analysisResults}
          isAnalyzing={isAnalyzing}
          onAnalyze={() => videoUrl && analyzeVideo(videoUrl, processedData?.metadata)}
        />
      )}

      {/* Analysis Results */}
      {analysisResults && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Analysis Results</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Sign Spots */}
              <div className="space-y-2">
                <h4 className="font-semibold text-sm text-gray-700">Detected Signs</h4>
                <div className="space-y-1">
                  {analysisResults.signSpots.map((spot, index) => (
                    <div key={index} className="flex items-center justify-between text-xs">
                      <span className="font-mono">{spot.gloss}</span>
                      <Badge variant="outline" className="text-xs">
                        {(spot.confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>

              {/* Hand Shapes */}
              <div className="space-y-2">
                <h4 className="font-semibold text-sm text-gray-700">Hand Shapes</h4>
                <div className="space-y-1">
                  {analysisResults.handShapes.map((shape, index) => (
                    <div key={index} className="flex items-center justify-between text-xs">
                      <span className="font-mono">{shape.shape}</span>
                      <Badge variant="outline" className="text-xs">
                        {(shape.confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>

              {/* Disambiguation */}
              <div className="space-y-2">
                <h4 className="font-semibold text-sm text-gray-700">Disambiguation</h4>
                <div className="text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span>Confidence:</span>
                    <Badge variant="outline" className="text-xs">
                      {(analysisResults.disambiguation.confidence * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <div className="bg-gray-50 p-2 rounded text-xs">
                    <div className="font-semibold">Final Sequence:</div>
                    <div className="font-mono">
                      {analysisResults.disambiguation.disambiguatedSequence.join(' ')}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 