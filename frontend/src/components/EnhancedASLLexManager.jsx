import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Download, 
  FileText, 
  Archive, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Clock, 
  RefreshCw,
  Eye,
  Search,
  Filter,
  BarChart3,
  Video,
  Tag,
  Hash,
  Calendar,
  FileSpreadsheet,
  Edit,
  TrendingUp,
  Database,
  Activity
} from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Progress } from './ui/progress';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { useToast } from './ui/use-toast';
import { 
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table';
import { 
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from './ui/tabs';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod';

const EnhancedASLLexManager = () => {
  const [signs, setSigns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSign, setSelectedSign] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [handshapeFilter, setHandshapeFilter] = useState('all');
  const [locationFilter, setLocationFilter] = useState('all');
  const [statistics, setStatistics] = useState({});
  const [viewMode, setViewMode] = useState('grid'); // grid, table, analytics
  const [selectedSigns, setSelectedSigns] = useState(new Set());
  const [bulkUploadJobs, setBulkUploadJobs] = useState([]);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [uploadedBy, setUploadedBy] = useState('');
  const [signTypes, setSignTypes] = useState([]);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [sortBy, setSortBy] = useState('uploaded_at');

  const { toast } = useToast();

  useEffect(() => {
    fetchSigns();
    fetchStatistics();
    fetchBulkUploadJobs();
    fetchSignTypes();
  }, []);

  useEffect(() => {
    fetchSigns();
  }, [searchTerm, statusFilter, handshapeFilter, locationFilter, sortBy]);

  const fetchSigns = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (statusFilter !== 'all') params.append('status', statusFilter);
      if (handshapeFilter !== 'all') params.append('handshape', handshapeFilter);
      if (locationFilter !== 'all') params.append('location', locationFilter);
      if (searchTerm) params.append('search', searchTerm);

      const response = await fetch(`${API_BASE_URL}/api/asl-lex/signs?${params}`);
      if (!response.ok) throw new Error('Failed to fetch signs');
      
      const data = await response.json();
      setSigns(data);
    } catch (err) {
      setError(err.message);
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/statistics`);
      if (!response.ok) throw new Error('Failed to fetch statistics');
      
      const data = await response.json();
      setStatistics(data);
    } catch (err) {
      console.error('Error fetching statistics:', err);
    }
  };

  const fetchBulkUploadJobs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/bulk-upload/jobs`);
      if (!response.ok) throw new Error('Failed to fetch bulk upload jobs');
      
      const data = await response.json();
      setBulkUploadJobs(data);
    } catch (err) {
      console.error('Error fetching bulk upload jobs:', err);
    }
  };

  const fetchSignTypes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/sign-types`);
      if (!response.ok) throw new Error('Failed to fetch sign types');
      
      const data = await response.json();
      setSignTypes(data);
    } catch (err) {
      console.error('Error fetching sign types:', err);
    }
  };

  const handleBulkUpload = async () => {
    if (!uploadFile) {
      toast({
        title: "Error",
        description: "Please select a file to upload",
        variant: "destructive",
      });
      return;
    }

    setUploadStatus('uploading');
    setUploadProgress(0);

    try {
      const uploadFormData = new FormData();
      uploadFormData.append('file', uploadFile);
      uploadFormData.append('uploaded_by', uploadedBy || 'data_analyst');

      const response = await fetch(`${API_BASE_URL}/api/asl-lex/bulk-upload`, {
        method: 'POST',
        body: uploadFormData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await response.json();
      
      setUploadStatus('completed');
      setUploadProgress(100);
      
      toast({
        title: "Success",
        description: "Bulk upload completed successfully!",
      });
      
      fetchBulkUploadJobs();
      fetchSigns();
      
      setTimeout(() => {
        setUploadStatus('idle');
        setUploadProgress(0);
      }, 3000);

    } catch (err) {
      setError(err.message);
      setUploadStatus('failed');
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    }
  };

  const downloadTemplate = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/bulk-upload/template`);
      if (!response.ok) throw new Error('Failed to download template');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'asl_lex_bulk_upload_template.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast({
        title: "Success",
        description: "Template downloaded successfully!",
      });
    } catch (err) {
      setError(err.message);
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800 border-green-200';
      case 'completed_with_errors': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'failed': return 'bg-red-100 text-red-800 border-red-200';
      case 'processing': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'cancelled': return 'bg-gray-100 text-gray-800 border-gray-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4" />;
      case 'completed_with_errors': return <AlertCircle className="h-4 w-4" />;
      case 'failed': return <XCircle className="h-4 w-4" />;
      case 'processing': return <RefreshCw className="h-4 w-4 animate-spin" />;
      case 'cancelled': return <XCircle className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  const handleSelectAll = () => {
    if (selectedSigns.size === signs.length) {
      setSelectedSigns(new Set());
    } else {
      setSelectedSigns(new Set(signs.map(s => s.id)));
    }
  };

  const handleSignSelection = (signId) => {
    const newSelected = new Set(selectedSigns);
    if (newSelected.has(signId)) {
      newSelected.delete(signId);
    } else {
      newSelected.add(signId);
    }
    setSelectedSigns(newSelected);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* SpokHand Logo */}
              <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl shadow-lg">
                <span className="text-white font-bold text-xl">SH</span>
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  SpokHand ASL-LEX
                </h1>
                <p className="text-gray-600 mt-2">
                  Advanced Sign Language Data Management System
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-gray-500">Powered by</span>
                  <span className="text-xs font-medium text-blue-600">SpokHand SLR</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="flex items-center gap-1">
                <Database className="h-3 w-3" />
                {statistics.total || 0} Signs
              </Badge>
              <Badge variant="outline" className="flex items-center gap-1">
                <Activity className="h-3 w-3" />
                {statistics.approved || 0} Approved
              </Badge>
            </div>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2">
                <Hash className="h-5 w-5" />
                Total Signs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{statistics.total || 0}</div>
              <p className="text-blue-100 text-sm">In database</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500 to-green-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5" />
                Approved
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{statistics.approved || 0}</div>
              <p className="text-green-100 text-sm">Validated signs</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-yellow-500 to-yellow-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Pending
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{statistics.pending || 0}</div>
              <p className="text-yellow-100 text-sm">Awaiting review</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500 to-purple-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Avg Confidence
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {statistics.avg_confidence ? Math.round(statistics.avg_confidence * 100) : 0}%
              </div>
              <p className="text-purple-100 text-sm">Quality score</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="signs" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="signs" className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Signs
            </TabsTrigger>
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="jobs" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Jobs
            </TabsTrigger>
          </TabsList>

          {/* Signs Tab */}
          <TabsContent value="signs" className="space-y-6">
            {/* Filters and Search */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Sign Management</CardTitle>
                    <CardDescription>
                      Browse, search, and manage ASL-LEX signs
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                    >
                      <Filter className="h-4 w-4 mr-2" />
                      Advanced Filters
                    </Button>
                    <Select value={viewMode} onValueChange={setViewMode}>
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="grid">Grid</SelectItem>
                        <SelectItem value="table">Table</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Search and Basic Filters */}
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                        <Input
                          placeholder="Search signs..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="pl-10"
                        />
                      </div>
                    </div>
                    <Select value={statusFilter} onValueChange={setStatusFilter}>
                      <SelectTrigger className="w-32">
                        <SelectValue placeholder="Status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="pending">Pending</SelectItem>
                        <SelectItem value="approved">Approved</SelectItem>
                        <SelectItem value="rejected">Rejected</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleSelectAll}
                    >
                      {selectedSigns.size === signs.length ? 'Deselect All' : 'Select All'}
                    </Button>
                  </div>

                  {/* Advanced Filters */}
                  {showAdvancedFilters && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
                      <Select value={handshapeFilter} onValueChange={setHandshapeFilter}>
                        <SelectTrigger>
                          <SelectValue placeholder="Handshape" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Handshapes</SelectItem>
                          {['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'].map(h => (
                            <SelectItem key={h} value={h}>{h}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Select value={locationFilter} onValueChange={setLocationFilter}>
                        <SelectTrigger>
                          <SelectValue placeholder="Location" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Locations</SelectItem>
                          <SelectItem value="neutral space">Neutral Space</SelectItem>
                          <SelectItem value="head">Head</SelectItem>
                          <SelectItem value="face">Face</SelectItem>
                          <SelectItem value="chest">Chest</SelectItem>
                          <SelectItem value="chin">Chin</SelectItem>
                        </SelectContent>
                      </Select>
                      <Select value={sortBy} onValueChange={setSortBy}>
                        <SelectTrigger>
                          <SelectValue placeholder="Sort by" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="uploaded_at">Upload Date</SelectItem>
                          <SelectItem value="gloss">Gloss</SelectItem>
                          <SelectItem value="status">Status</SelectItem>
                          <SelectItem value="confidence_score">Confidence</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Signs Display */}
            {loading ? (
              <Card>
                <CardContent className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <RefreshCw className="h-8 w-8 text-blue-600 animate-spin mx-auto mb-2" />
                    <p className="text-gray-600">Loading signs...</p>
                  </div>
                </CardContent>
              </Card>
            ) : signs.length === 0 ? (
              <Card>
                <CardContent className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <FileText className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-600">No signs found</p>
                    <p className="text-sm text-gray-500 mt-1">
                      Upload some signs to get started
                    </p>
                  </div>
                </CardContent>
              </Card>
            ) : viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {signs.map((sign) => (
                  <Card key={sign.id} className="hover:shadow-lg transition-shadow duration-200">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={selectedSigns.has(sign.id)}
                            onChange={() => handleSignSelection(sign.id)}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          />
                          <div>
                            <CardTitle className="text-lg">{sign.gloss}</CardTitle>
                            <CardDescription>{sign.english}</CardDescription>
                          </div>
                        </div>
                        <Badge 
                          variant={sign.status === 'approved' ? 'default' : 'secondary'}
                          className={sign.status === 'approved' ? 'bg-green-100 text-green-800' : ''}
                        >
                          {sign.status}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="flex items-center gap-1">
                          <Hash className="h-3 w-3 text-gray-400" />
                          <span className="text-gray-600">Handshape: {sign.handshape}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Tag className="h-3 w-3 text-gray-400" />
                          <span className="text-gray-600">Location: {sign.location}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 text-sm">
                        <Calendar className="h-3 w-3 text-gray-400" />
                        <span className="text-gray-600">
                          {new Date(sign.uploaded_at).toLocaleDateString()}
                        </span>
                      </div>
                      {sign.video_url && (
                        <div className="flex items-center gap-1 text-sm">
                          <Video className="h-3 w-3 text-gray-400" />
                          <span className="text-gray-600">Video available</span>
                        </div>
                      )}
                      <div className="flex gap-2 pt-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedSign(sign)}
                          className="flex-1"
                        >
                          <Eye className="h-3 w-3 mr-1" />
                          View
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1"
                        >
                          <Edit className="h-3 w-3 mr-1" />
                          Edit
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12">
                          <input
                            type="checkbox"
                            checked={selectedSigns.size === signs.length}
                            onChange={handleSelectAll}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          />
                        </TableHead>
                        <TableHead>Gloss</TableHead>
                        <TableHead>English</TableHead>
                        <TableHead>Handshape</TableHead>
                        <TableHead>Location</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Upload Date</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {signs.map((sign) => (
                        <TableRow key={sign.id}>
                          <TableCell>
                            <input
                              type="checkbox"
                              checked={selectedSigns.has(sign.id)}
                              onChange={() => handleSignSelection(sign.id)}
                              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                            />
                          </TableCell>
                          <TableCell className="font-medium">{sign.gloss}</TableCell>
                          <TableCell>{sign.english}</TableCell>
                          <TableCell>{sign.handshape}</TableCell>
                          <TableCell>{sign.location}</TableCell>
                          <TableCell>
                            <Badge 
                              variant={sign.status === 'approved' ? 'default' : 'secondary'}
                              className={sign.status === 'approved' ? 'bg-green-100 text-green-800' : ''}
                            >
                              {sign.status}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {new Date(sign.uploaded_at).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            <div className="flex gap-1">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setSelectedSign(sign)}
                              >
                                <Eye className="h-3 w-3" />
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                              >
                                <Edit className="h-3 w-3" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Upload Tab */}
          <TabsContent value="upload" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Individual Upload */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Video className="h-5 w-5" />
                    Individual Upload
                  </CardTitle>
                  <CardDescription>
                    Upload a single video with metadata
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                    <input
                      type="file"
                      accept="video/*"
                      onChange={(e) => setUploadFile(e.target.files[0])}
                      className="hidden"
                      id="video-upload"
                    />
                    <label htmlFor="video-upload" className="cursor-pointer">
                      <Video className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">
                        {uploadFile ? uploadFile.name : 'Click to select video file'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Supports MP4, AVI, MOV, WebM formats
                      </p>
                    </label>
                  </div>
                  
                  <div className="space-y-3">
                    <Input placeholder="Gloss (e.g., HELLO)" />
                    <Input placeholder="English Translation" />
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Sign Type" />
                      </SelectTrigger>
                      <SelectContent>
                        {signTypes.map((type) => (
                          <SelectItem key={type.value} value={type.value}>
                            {type.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Input placeholder="Uploaded By" />
                  </div>

                  <Button className="w-full" disabled={!uploadFile}>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Video
                  </Button>
                </CardContent>
              </Card>

              {/* Bulk Upload */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Archive className="h-5 w-5" />
                    Bulk Upload
                  </CardTitle>
                  <CardDescription>
                    Upload CSV or ZIP files with multiple signs
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                    <input
                      type="file"
                      accept=".csv,.zip"
                      onChange={(e) => setUploadFile(e.target.files[0])}
                      className="hidden"
                      id="bulk-upload"
                    />
                    <label htmlFor="bulk-upload" className="cursor-pointer">
                      <FileSpreadsheet className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">
                        {uploadFile ? uploadFile.name : 'Click to select file'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Supports CSV files or ZIP files with videos
                      </p>
                    </label>
                  </div>

                  <div className="space-y-3">
                    <Input 
                      placeholder="Uploaded By" 
                      value={uploadedBy}
                      onChange={(e) => setUploadedBy(e.target.value)}
                    />
                    
                    {uploadStatus !== 'idle' && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>Upload Progress</span>
                          <span>{uploadProgress}%</span>
                        </div>
                        <Progress value={uploadProgress} className="h-2" />
                      </div>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <Button 
                      className="flex-1" 
                      disabled={!uploadFile || uploadStatus === 'uploading'}
                      onClick={handleBulkUpload}
                    >
                      {uploadStatus === 'uploading' ? (
                        <>
                          <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                          Uploading...
                        </>
                      ) : (
                        <>
                          <Upload className="h-4 w-4 mr-2" />
                          Upload
                        </>
                      )}
                    </Button>
                    <Button variant="outline" onClick={downloadTemplate}>
                      <Download className="h-4 w-4 mr-2" />
                      Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Sign Type Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(statistics.handshapes || {}).map(([handshape, count]) => (
                      <div key={handshape} className="flex items-center justify-between">
                        <span className="font-medium">{handshape}</span>
                        <Badge variant="secondary">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Upload Trends
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span>Total Signs</span>
                      <span className="font-bold">{statistics.total || 0}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Approved</span>
                      <span className="font-bold text-green-600">{statistics.approved || 0}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Pending</span>
                      <span className="font-bold text-yellow-600">{statistics.pending || 0}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Rejected</span>
                      <span className="font-bold text-red-600">{statistics.rejected || 0}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Jobs Tab */}
          <TabsContent value="jobs" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Bulk Upload Jobs
                </CardTitle>
                <CardDescription>
                  Monitor and manage bulk upload operations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {bulkUploadJobs.length === 0 ? (
                  <div className="text-center py-8">
                    <Activity className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500">No bulk upload jobs found</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {bulkUploadJobs.map((job) => (
                      <Card key={job.id} className="border-l-4 border-l-blue-500">
                        <CardContent className="p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              {job.file_type === 'csv' ? (
                                <FileSpreadsheet className="h-5 w-5 text-blue-600" />
                              ) : (
                                <Archive className="h-5 w-5 text-purple-600" />
                              )}
                              <div>
                                <h3 className="font-medium">{job.filename}</h3>
                                <p className="text-sm text-gray-500">
                                  {job.total_items} items • {job.successful_items} successful • {job.failed_items} failed
                                </p>
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              <Badge className={getStatusColor(job.status)}>
                                <div className="flex items-center gap-1">
                                  {getStatusIcon(job.status)}
                                  {job.status}
                                </div>
                              </Badge>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
};

export default EnhancedASLLexManager; 