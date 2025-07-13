import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { 
  Play, 
  Download, 
  Search, 
  Filter, 
  Eye, 
  FileText, 
  Video,
  Database,
  ExternalLink
} from 'lucide-react';

export default function DatasetViewer() {
  const [wlaslVideos, setWlaslVideos] = useState([]);
  const [aslLexFiles, setAslLexFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSplit, setSelectedSplit] = useState('all');
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [selectedCsvFile, setSelectedCsvFile] = useState(null);
  const [csvData, setCsvData] = useState(null);

  // Load datasets on component mount
  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    setLoading(true);
    try {
      // Load WLASL videos
      const videosResponse = await fetch('/api/wlasl-videos');
      if (videosResponse.ok) {
        const videos = await videosResponse.json();
        setWlaslVideos(videos);
      }

      // Load ASL-LEX files
      const csvResponse = await fetch('/api/asl-lex-files');
      if (csvResponse.ok) {
        const files = await csvResponse.json();
        setAslLexFiles(files);
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredVideos = wlaslVideos.filter(video => {
    const matchesSearch = video.gloss?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         video.filename?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSplit = selectedSplit === 'all' || video.split === selectedSplit;
    return matchesSearch && matchesSplit;
  });

  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
    setSelectedCsvFile(null);
    setCsvData(null);
  };

  const handleCsvFileSelect = async (file) => {
    setSelectedCsvFile(file);
    setSelectedVideo(null);
    
    try {
      const response = await fetch(`/api/asl-lex-data/${file.file_id}`);
      if (response.ok) {
        const data = await response.json();
        setCsvData(data);
      }
    } catch (error) {
      console.error('Error loading CSV data:', error);
    }
  };

  const generateStreamingUrl = async (videoId) => {
    try {
      const response = await fetch(`/api/wlasl-videos/${videoId}/stream-url`, {
        method: 'POST'
      });
      if (response.ok) {
        const { presigned_url } = await response.json();
        return presigned_url;
      }
    } catch (error) {
      console.error('Error generating streaming URL:', error);
    }
    return null;
  };

  const handlePlayVideo = async (video) => {
    const streamingUrl = await generateStreamingUrl(video.video_id);
    if (streamingUrl) {
      window.open(streamingUrl, '_blank');
    }
  };

  const handleDownloadCsv = async (file) => {
    try {
      const response = await fetch(`/api/asl-lex-files/${file.file_id}/download`);
      if (response.ok) {
        const { presigned_url } = await response.json();
        window.open(presigned_url, '_blank');
      }
    } catch (error) {
      console.error('Error downloading CSV:', error);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Dataset Viewer</h1>
        <p className="text-muted-foreground">
          View and interact with your uploaded WLASL videos and ASL-LEX CSV files
        </p>
      </div>

      <Tabs defaultValue="videos" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="videos" className="flex items-center gap-2">
            <Video className="h-4 w-4" />
            WLASL Videos ({wlaslVideos.length})
          </TabsTrigger>
          <TabsTrigger value="csv" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            ASL-LEX CSV ({aslLexFiles.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="videos" className="space-y-6">
          {/* Search and Filter Controls */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <Label htmlFor="search">Search Videos</Label>
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="search"
                  placeholder="Search by gloss or filename..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
            <div className="sm:w-48">
              <Label htmlFor="split">Split</Label>
              <select
                id="split"
                value={selectedSplit}
                onChange={(e) => setSelectedSplit(e.target.value)}
                className="w-full px-3 py-2 border border-input rounded-md"
              >
                <option value="all">All Splits</option>
                <option value="train">Train</option>
                <option value="val">Validation</option>
                <option value="test">Test</option>
              </select>
            </div>
          </div>

          {/* Video Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredVideos.map((video) => (
              <Card 
                key={video.video_id} 
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedVideo?.video_id === video.video_id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => handleVideoSelect(video)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="text-sm font-medium truncate">
                        {video.gloss || 'Unknown Gloss'}
                      </CardTitle>
                      <CardDescription className="text-xs">
                        {video.filename}
                      </CardDescription>
                    </div>
                    <Badge variant={video.split === 'train' ? 'default' : 'secondary'}>
                      {video.split}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-muted-foreground">
                      {video.file_format} • {video.file_size ? `${(video.file_size / 1024 / 1024).toFixed(1)}MB` : 'Unknown size'}
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePlayVideo(video);
                      }}
                    >
                      <Play className="h-3 w-3 mr-1" />
                      Play
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Selected Video Details */}
          {selectedVideo && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Video Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm font-medium">Gloss</Label>
                    <p className="text-sm text-muted-foreground">{selectedVideo.gloss}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Split</Label>
                    <Badge variant={selectedVideo.split === 'train' ? 'default' : 'secondary'}>
                      {selectedVideo.split}
                    </Badge>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">File Format</Label>
                    <p className="text-sm text-muted-foreground">{selectedVideo.file_format}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Uploaded</Label>
                    <p className="text-sm text-muted-foreground">
                      {new Date(selectedVideo.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <div className="mt-4 flex gap-2">
                  <Button onClick={() => handlePlayVideo(selectedVideo)}>
                    <Play className="h-4 w-4 mr-2" />
                    Play Video
                  </Button>
                  <Button variant="outline">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View in S3
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="csv" className="space-y-6">
          {/* CSV Files Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {aslLexFiles.map((file) => (
              <Card 
                key={file.file_id} 
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedCsvFile?.file_id === file.file_id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => handleCsvFileSelect(file)}
              >
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium truncate">
                    {file.filename}
                  </CardTitle>
                  <CardDescription className="text-xs">
                    {file.description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-muted-foreground">
                      {file.row_count} rows • {file.column_count} columns
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDownloadCsv(file);
                      }}
                    >
                      <Download className="h-3 w-3 mr-1" />
                      Download
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Selected CSV Details */}
          {selectedCsvFile && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>CSV File Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div>
                    <Label className="text-sm font-medium">Filename</Label>
                    <p className="text-sm text-muted-foreground">{selectedCsvFile.filename}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Data Type</Label>
                    <Badge variant="outline">{selectedCsvFile.data_type || 'general_data'}</Badge>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Rows</Label>
                    <p className="text-sm text-muted-foreground">{selectedCsvFile.row_count}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Columns</Label>
                    <p className="text-sm text-muted-foreground">{selectedCsvFile.column_count}</p>
                  </div>
                </div>
                
                {csvData && (
                  <div className="mt-4">
                    <Label className="text-sm font-medium">Columns</Label>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {csvData.columns.map((col, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {col}
                        </Badge>
                      ))}
                    </div>
                    
                    <div className="mt-4">
                      <Label className="text-sm font-medium">Sample Data (First 5 rows)</Label>
                      <div className="mt-2 max-h-60 overflow-auto border rounded-md">
                        <table className="w-full text-xs">
                          <thead className="bg-muted">
                            <tr>
                              {csvData.columns.map((col, index) => (
                                <th key={index} className="px-2 py-1 text-left">{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {csvData.data.slice(0, 5).map((row, rowIndex) => (
                              <tr key={rowIndex} className="border-t">
                                {csvData.columns.map((col, colIndex) => (
                                  <td key={colIndex} className="px-2 py-1">
                                    {String(row[col] || '').substring(0, 50)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="mt-4 flex gap-2">
                  <Button onClick={() => handleDownloadCsv(selectedCsvFile)}>
                    <Download className="h-4 w-4 mr-2" />
                    Download CSV
                  </Button>
                  <Button variant="outline">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View in S3
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
} 