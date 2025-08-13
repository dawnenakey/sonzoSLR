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
  Trash2,
  Eye,
  Plus,
  Search,
  Filter,
  BarChart3,
  Settings,
  Users,
  Video,
  Tag,
  Hash,
  Calendar,
  User,
  FileSpreadsheet,
  FileVideo,
  CheckSquare,
  Square,
  Edit,
  X,
  AlertTriangle,
  Info
} from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
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

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod';

const ASLLexDataManager = () => {
  const [signs, setSigns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSign, setSelectedSign] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [handshapeFilter, setHandshapeFilter] = useState('all');
  const [locationFilter, setLocationFilter] = useState('all');
  const [statistics, setStatistics] = useState({});
  const [showStatistics, setShowStatistics] = useState(false);
  
  // Bulk upload states
  const [bulkUploadJobs, setBulkUploadJobs] = useState([]);
  const [showBulkUpload, setShowBulkUpload] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [uploadedBy, setUploadedBy] = useState('');
  const [showJobDetails, setShowJobDetails] = useState(null);
  const [pollingJobs, setPollingJobs] = useState(new Set());

  // Individual video upload states
  const [showIndividualUpload, setShowIndividualUpload] = useState(false);
  const [individualVideoFile, setIndividualVideoFile] = useState(null);
  const [signTypes, setSignTypes] = useState([]);
  const [individualUploadData, setIndividualUploadData] = useState({
    gloss: '',
    english: '',
    sign_type: 'isolated_sign',
    handshape: '',
    location: '',
    movement: '',
    palm_orientation: '',
    dominant_hand: '',
    non_dominant_hand: '',
    frequency: 0,
    age_of_acquisition: 0,
    iconicity: 0,
    lexical_class: '',
    tags: '',
    notes: '',
    confidence_score: 0.8
  });

  // Video preview states
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [showVideoPreview, setShowVideoPreview] = useState(false);

  // Custom sign type states
  const [showCustomSignTypeForm, setShowCustomSignTypeForm] = useState(false);
  const [customSignTypeData, setCustomSignTypeData] = useState({
    custom_type: '',
    description: ''
  });

  // Batch update states
  const [selectedSigns, setSelectedSigns] = useState(new Set());
  const [showBatchUpdate, setShowBatchUpdate] = useState(false);
  const [batchSignType, setBatchSignType] = useState('isolated_sign');

  // ASL validation states
  const [validationResults, setValidationResults] = useState(null);
  const [showValidation, setShowValidation] = useState(false);

  // Analytics states
  const [signTypeAnalytics, setSignTypeAnalytics] = useState(null);
  const [showAnalytics, setShowAnalytics] = useState(false);

  // Filter states
  const [signTypeFilter, setSignTypeFilter] = useState('all');

  useEffect(() => {
    fetchSigns();
    fetchStatistics();
    fetchBulkUploadJobs();
    fetchSignTypes();
  }, []);

  useEffect(() => {
    fetchSigns();
  }, [searchTerm, statusFilter, handshapeFilter, locationFilter, signTypeFilter]);

  // Poll for job updates
  useEffect(() => {
    if (pollingJobs.size > 0) {
      const interval = setInterval(() => {
        pollingJobs.forEach(jobId => {
          fetchBulkUploadJob(jobId);
        });
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [pollingJobs]);

  const fetchSigns = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (statusFilter !== 'all') params.append('status', statusFilter);
      if (handshapeFilter !== 'all') params.append('handshape', handshapeFilter);
      if (locationFilter !== 'all') params.append('location', locationFilter);
      if (signTypeFilter !== 'all') params.append('sign_type', signTypeFilter);
      if (searchTerm) params.append('search', searchTerm);

      const response = await fetch(`${API_BASE_URL}/api/asl-lex/signs?${params}`);
      if (!response.ok) throw new Error('Failed to fetch signs');
      
      const data = await response.json();
      setSigns(data);
    } catch (err) {
      setError(err.message);
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

  const fetchBulkUploadJob = async (jobId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/bulk-upload/jobs/${jobId}`);
      if (!response.ok) throw new Error('Failed to fetch job details');
      
      const job = await response.json();
      
      // Update job in the list
      setBulkUploadJobs(prev => prev.map(j => j.id === jobId ? job : j));
      
      // Stop polling if job is completed or failed
      if (['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(job.status)) {
        setPollingJobs(prev => {
          const newSet = new Set(prev);
          newSet.delete(jobId);
          return newSet;
        });
      }
    } catch (err) {
      console.error('Error fetching job details:', err);
    }
  };

  const handleBulkUpload = async () => {
    if (!uploadFile) {
      setError('Please select a file to upload');
      return;
    }

    setUploadStatus('uploading');
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', uploadFile);
      formData.append('uploaded_by', uploadedBy || 'data_analyst');

      const response = await fetch(`${API_BASE_URL}/api/asl-lex/bulk-upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await response.json();
      
      // Start polling for job updates
      setPollingJobs(prev => new Set([...prev, result.job_id]));
      
      setUploadStatus('completed');
      setUploadProgress(100);
      
      // Refresh jobs list
      fetchBulkUploadJobs();
      
      // Reset form
      setUploadFile(null);
      setUploadedBy('');
      
      setTimeout(() => {
        setUploadStatus('idle');
        setUploadProgress(0);
      }, 3000);

    } catch (err) {
      setError(err.message);
      setUploadStatus('failed');
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
    } catch (err) {
      setError(err.message);
    }
  };

  const fetchSignTypes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/sign-types`);
      if (!response.ok) throw new Error('Failed to fetch sign types');
      
      const data = await response.json();
      
      // Fetch custom sign types
      const customResponse = await fetch(`${API_BASE_URL}/api/asl-lex/sign-types/custom`);
      if (customResponse.ok) {
        const customData = await customResponse.json();
        setSignTypes([...data, ...customData]);
      } else {
        setSignTypes(data);
      }
    } catch (err) {
      console.error('Error fetching sign types:', err);
    }
  };

  const handleVideoPreview = (file) => {
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoPreviewUrl(url);
      setShowVideoPreview(true);
    }
  };

  const addCustomSignType = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/sign-types`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          custom_type: customSignTypeData.custom_type,
          description: customSignTypeData.description,
          created_by: uploadedBy || 'data_analyst'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to add custom sign type');
      }

      const result = await response.json();
      
      // Refresh sign types
      fetchSignTypes();
      
      // Reset form
      setCustomSignTypeData({ custom_type: '', description: '' });
      setShowCustomSignTypeForm(false);
      
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const validateASLSign = async (signData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/validate-asl-sign`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(signData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Validation failed');
      }

      const results = await response.json();
      setValidationResults(results);
      setShowValidation(true);
      
      return results;
    } catch (err) {
      setError(err.message);
      return null;
    }
  };

  const batchUpdateSignTypes = async () => {
    if (selectedSigns.size === 0) {
      setError('Please select at least one sign to update');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/signs/batch-update-type`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sign_ids: Array.from(selectedSigns),
          sign_type: batchSignType,
          updated_by: uploadedBy || 'data_analyst'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Batch update failed');
      }

      const result = await response.json();
      
      // Refresh signs list
      fetchSigns();
      
      // Clear selection
      setSelectedSigns(new Set());
      setShowBatchUpdate(false);
      
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const fetchSignTypeAnalytics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/analytics/sign-types`);
      if (!response.ok) throw new Error('Failed to fetch analytics');
      
      const data = await response.json();
      setSignTypeAnalytics(data);
      setShowAnalytics(true);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleIndividualVideoUpload = async () => {
    if (!individualVideoFile) {
      setError('Please select a video file');
      return;
    }

    if (!individualUploadData.gloss || !individualUploadData.english) {
      setError('Gloss and English translation are required');
      return;
    }

    setUploadStatus('uploading');
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('video', individualVideoFile);
      formData.append('uploaded_by', uploadedBy || 'data_analyst');
      
      // Add all form data
      Object.keys(individualUploadData).forEach(key => {
        formData.append(key, individualUploadData[key]);
      });

      const response = await fetch(`${API_BASE_URL}/api/asl-lex/upload-video-with-metadata`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await response.json();
      
      setUploadStatus('completed');
      setUploadProgress(100);
      
      // Reset form
      setIndividualVideoFile(null);
      setIndividualUploadData({
        gloss: '',
        english: '',
        sign_type: 'isolated_sign',
        handshape: '',
        location: '',
        movement: '',
        palm_orientation: '',
        dominant_hand: '',
        non_dominant_hand: '',
        frequency: 0,
        age_of_acquisition: 0,
        iconicity: 0,
        lexical_class: '',
        tags: '',
        notes: '',
        confidence_score: 0.8
      });
      
      // Refresh signs list
      fetchSigns();
      
      setTimeout(() => {
        setUploadStatus('idle');
        setUploadProgress(0);
      }, 3000);

    } catch (err) {
      setError(err.message);
      setUploadStatus('failed');
    }
  };

  const cancelJob = async (jobId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/asl-lex/bulk-upload/jobs/${jobId}/cancel`, {
        method: 'POST',
      });
      
      if (!response.ok) throw new Error('Failed to cancel job');
      
      // Refresh jobs list
      fetchBulkUploadJobs();
    } catch (err) {
      setError(err.message);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'completed_with_errors': return 'text-yellow-600 bg-yellow-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'processing': return 'text-blue-600 bg-blue-100';
      case 'cancelled': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
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

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">ASL-LEX Data Management</h1>
        <p className="text-gray-600">Manage ASL-LEX sign data, upload videos, and validate entries for training the SLR software.</p>
      </div>

      {/* Statistics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Hash className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Signs</p>
              <p className="text-2xl font-bold text-gray-900">{statistics.total || 0}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-green-100 rounded-lg">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Approved</p>
              <p className="text-2xl font-bold text-gray-900">{statistics.approved || 0}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <Clock className="h-6 w-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Pending</p>
              <p className="text-2xl font-bold text-gray-900">{statistics.pending || 0}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <BarChart3 className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Confidence</p>
              <p className="text-2xl font-bold text-gray-900">
                {statistics.avg_confidence ? Math.round(statistics.avg_confidence * 100) : 0}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-sm border mb-8">
        <div className="p-6 border-b">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">Video Upload</h2>
              <p className="text-gray-600">Upload individual videos with metadata or bulk upload CSV/ZIP files</p>
            </div>
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => setShowIndividualUpload(!showIndividualUpload)}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-green-600 bg-green-50 rounded-lg hover:bg-green-100"
              >
                <Video className="h-4 w-4" />
                Individual Upload
              </button>
              <button
                onClick={() => setShowCustomSignTypeForm(true)}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-orange-600 bg-orange-50 rounded-lg hover:bg-orange-100"
              >
                <Plus className="h-4 w-4" />
                Add Sign Type
              </button>
              <button
                onClick={() => setShowBatchUpdate(true)}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-indigo-600 bg-indigo-50 rounded-lg hover:bg-indigo-100"
                disabled={selectedSigns.size === 0}
              >
                <Edit className="h-4 w-4" />
                Batch Update ({selectedSigns.size})
              </button>
              <button
                onClick={fetchSignTypeAnalytics}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-teal-600 bg-teal-50 rounded-lg hover:bg-teal-100"
              >
                <BarChart3 className="h-4 w-4" />
                Analytics
              </button>
              <button
                onClick={downloadTemplate}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100"
              >
                <Download className="h-4 w-4" />
                Download Template
              </button>
              <button
                onClick={() => setShowBulkUpload(!showBulkUpload)}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
              >
                <Upload className="h-4 w-4" />
                Bulk Upload
              </button>
            </div>
          </div>
        </div>

        {/* Individual Video Upload */}
        {showIndividualUpload && (
          <div className="p-6 border-b">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Video Upload */}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload Video File
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                    <input
                      type="file"
                      accept="video/*"
                      onChange={(e) => setIndividualVideoFile(e.target.files[0])}
                      className="hidden"
                      id="individual-video-upload"
                    />
                                         <label htmlFor="individual-video-upload" className="cursor-pointer">
                       <Video className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                       <p className="text-sm text-gray-600">
                         {individualVideoFile ? individualVideoFile.name : 'Click to select video file'}
                       </p>
                       <p className="text-xs text-gray-500 mt-1">
                         Supports MP4, AVI, MOV, WebM formats
                       </p>
                     </label>
                   </div>
                   
                   {individualVideoFile && (
                     <div className="space-y-2">
                       <button
                         onClick={() => handleVideoPreview(individualVideoFile)}
                         className="flex items-center gap-2 px-3 py-1 text-sm text-blue-600 hover:text-blue-800"
                       >
                         <Eye className="h-4 w-4" />
                         Preview Video
                       </button>
                       
                       <button
                         onClick={() => validateASLSign(individualUploadData)}
                         className="flex items-center gap-2 px-3 py-1 text-sm text-green-600 hover:text-green-800"
                       >
                         <CheckCircle className="h-4 w-4" />
                         Validate ASL Data
                       </button>
                     </div>
                   )}
                 </div>

                {/* Basic Information */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Gloss (ASL Sign Name) *
                    </label>
                    <input
                      type="text"
                      value={individualUploadData.gloss}
                      onChange={(e) => setIndividualUploadData(prev => ({ ...prev, gloss: e.target.value }))}
                      placeholder="e.g., HELLO"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      English Translation *
                    </label>
                    <input
                      type="text"
                      value={individualUploadData.english}
                      onChange={(e) => setIndividualUploadData(prev => ({ ...prev, english: e.target.value }))}
                      placeholder="e.g., Hello"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Sign Type *
                    </label>
                    <select
                      value={individualUploadData.sign_type}
                      onChange={(e) => setIndividualUploadData(prev => ({ ...prev, sign_type: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {signTypes.map((type) => (
                        <option key={type.value} value={type.value}>
                          {type.label} - {type.description}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Uploaded By
                    </label>
                    <input
                      type="text"
                      value={uploadedBy}
                      onChange={(e) => setUploadedBy(e.target.value)}
                      placeholder="Enter your name or ID"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Advanced Metadata */}
              <div className="space-y-4">
                <h3 className="font-medium text-gray-900 mb-3">Advanced Metadata (Optional)</h3>
                
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Handshape</label>
                    <select
                      value={individualUploadData.handshape}
                      onChange={(e) => setIndividualUploadData(prev => ({ ...prev, handshape: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select handshape</option>
                      {['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'].map(h => (
                        <option key={h} value={h}>{h}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                    <select
                      value={individualUploadData.location}
                      onChange={(e) => setIndividualUploadData(prev => ({ ...prev, location: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select location</option>
                      <option value="neutral space">Neutral Space</option>
                      <option value="head">Head</option>
                      <option value="face">Face</option>
                      <option value="chest">Chest</option>
                      <option value="waist">Waist</option>
                      <option value="chin">Chin</option>
                      <option value="forehead">Forehead</option>
                      <option value="nose">Nose</option>
                      <option value="mouth">Mouth</option>
                      <option value="ear">Ear</option>
                      <option value="eye">Eye</option>
                      <option value="cheek">Cheek</option>
                      <option value="shoulder">Shoulder</option>
                      <option value="arm">Arm</option>
                      <option value="hand">Hand</option>
                      <option value="leg">Leg</option>
                      <option value="foot">Foot</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Movement</label>
                  <input
                    type="text"
                    value={individualUploadData.movement}
                    onChange={(e) => setIndividualUploadData(prev => ({ ...prev, movement: e.target.value }))}
                    placeholder="e.g., wave, forward, downward"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Lexical Class</label>
                  <select
                    value={individualUploadData.lexical_class}
                    onChange={(e) => setIndividualUploadData(prev => ({ ...prev, lexical_class: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select lexical class</option>
                    <option value="noun">Noun</option>
                    <option value="verb">Verb</option>
                    <option value="adjective">Adjective</option>
                    <option value="adverb">Adverb</option>
                    <option value="interjection">Interjection</option>
                    <option value="pronoun">Pronoun</option>
                    <option value="conjunction">Conjunction</option>
                    <option value="preposition">Preposition</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tags</label>
                  <input
                    type="text"
                    value={individualUploadData.tags}
                    onChange={(e) => setIndividualUploadData(prev => ({ ...prev, tags: e.target.value }))}
                    placeholder="comma-separated tags"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
                  <textarea
                    value={individualUploadData.notes}
                    onChange={(e) => setIndividualUploadData(prev => ({ ...prev, notes: e.target.value }))}
                    placeholder="Additional notes about this sign..."
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {uploadStatus !== 'idle' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">Upload Progress</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600">
                      {uploadStatus === 'uploading' && 'Uploading video and metadata...'}
                      {uploadStatus === 'completed' && 'Upload completed successfully!'}
                      {uploadStatus === 'failed' && 'Upload failed. Please try again.'}
                    </p>
                  </div>
                )}

                <button
                  onClick={handleIndividualVideoUpload}
                  disabled={!individualVideoFile || !individualUploadData.gloss || !individualUploadData.english || uploadStatus === 'uploading'}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploadStatus === 'uploading' ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Video className="h-4 w-4" />
                      Upload Video with Metadata
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {showBulkUpload && (
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Upload Form */}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload File (CSV or ZIP)
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                    <input
                      type="file"
                      accept=".csv,.zip"
                      onChange={(e) => setUploadFile(e.target.files[0])}
                      className="hidden"
                      id="bulk-upload-file"
                    />
                    <label htmlFor="bulk-upload-file" className="cursor-pointer">
                      <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">
                        {uploadFile ? uploadFile.name : 'Click to select file'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Supports CSV files or ZIP files with videos and metadata
                      </p>
                    </label>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Uploaded By
                  </label>
                  <input
                    type="text"
                    value={uploadedBy}
                    onChange={(e) => setUploadedBy(e.target.value)}
                    placeholder="Enter your name or ID"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {uploadStatus !== 'idle' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">Upload Progress</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600">
                      {uploadStatus === 'uploading' && 'Uploading and processing...'}
                      {uploadStatus === 'completed' && 'Upload completed successfully!'}
                      {uploadStatus === 'failed' && 'Upload failed. Please try again.'}
                    </p>
                  </div>
                )}

                <button
                  onClick={handleBulkUpload}
                  disabled={!uploadFile || uploadStatus === 'uploading'}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploadStatus === 'uploading' ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      Upload and Process
                    </>
                  )}
                </button>
              </div>

              {/* Instructions */}
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-medium text-blue-900 mb-2">CSV Upload Instructions</h3>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Download the template CSV file</li>
                    <li>• Fill in sign metadata (gloss, english, handshape, etc.)</li>
                    <li>• Include video_filename column for video references</li>
                    <li>• Upload the CSV file</li>
                  </ul>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <h3 className="font-medium text-green-900 mb-2">ZIP Upload Instructions</h3>
                  <ul className="text-sm text-green-800 space-y-1">
                    <li>• Create a ZIP file with videos and CSV metadata</li>
                    <li>• Include one CSV file with sign metadata</li>
                    <li>• Include video files referenced in the CSV</li>
                    <li>• Upload the ZIP file for automatic processing</li>
                  </ul>
                </div>

                <div className="bg-yellow-50 p-4 rounded-lg">
                  <h3 className="font-medium text-yellow-900 mb-2">Required Fields</h3>
                  <ul className="text-sm text-yellow-800 space-y-1">
                    <li>• gloss: ASL gloss (e.g., "HELLO")</li>
                    <li>• english: English translation</li>
                    <li>• handshape: Hand shape classification</li>
                    <li>• location: Sign location</li>
                    <li>• movement: Movement description</li>
                    <li>• video_filename: Video file name (for ZIP uploads)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bulk Upload Jobs */}
      <div className="bg-white rounded-lg shadow-sm border mb-8">
        <div className="p-6 border-b">
          <h2 className="text-xl font-semibold text-gray-900">Bulk Upload Jobs</h2>
        </div>
        <div className="p-6">
          {bulkUploadJobs.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No bulk upload jobs found</p>
          ) : (
            <div className="space-y-4">
              {bulkUploadJobs.map((job) => (
                <div key={job.id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {job.file_type === 'csv' ? (
                        <FileSpreadsheet className="h-5 w-5 text-blue-600" />
                      ) : (
                        <Archive className="h-5 w-5 text-purple-600" />
                      )}
                      <div>
                        <h3 className="font-medium text-gray-900">{job.filename}</h3>
                        <p className="text-sm text-gray-500">
                          {job.total_items} items • {job.successful_items} successful • {job.failed_items} failed
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                        <div className="flex items-center gap-1">
                          {getStatusIcon(job.status)}
                          {job.status}
                        </div>
                      </span>
                      
                      <button
                        onClick={() => setShowJobDetails(showJobDetails === job.id ? null : job.id)}
                        className="p-1 text-gray-400 hover:text-gray-600"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      
                      {job.status === 'processing' && (
                        <button
                          onClick={() => cancelJob(job.id)}
                          className="p-1 text-red-400 hover:text-red-600"
                        >
                          <XCircle className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                  </div>

                  {showJobDetails === job.id && (
                    <div className="mt-4 pt-4 border-t">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-gray-500">Uploaded By</p>
                          <p className="font-medium">{job.uploaded_by}</p>
                        </div>
                        <div>
                          <p className="text-gray-500">Upload Date</p>
                          <p className="font-medium">
                            {new Date(job.uploaded_at).toLocaleDateString()}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-500">Progress</p>
                          <p className="font-medium">
                            {job.processed_items} / {job.total_items}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-500">Success Rate</p>
                          <p className="font-medium">
                            {job.total_items > 0 ? Math.round((job.successful_items / job.total_items) * 100) : 0}%
                          </p>
                        </div>
                      </div>

                      {job.error_log && job.error_log.length > 0 && (
                        <div className="mt-4">
                          <p className="text-sm font-medium text-gray-700 mb-2">Errors:</p>
                          <div className="bg-red-50 p-3 rounded-lg max-h-32 overflow-y-auto">
                            {job.error_log.map((error, index) => (
                              <p key={index} className="text-sm text-red-700">
                                {error}
                              </p>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <p className="text-red-800">{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="mt-2 text-sm text-red-600 hover:text-red-800"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Existing ASL-LEX Management UI */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">Sign Management</h2>
            <button
              onClick={() => setShowForm(!showForm)}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
            >
              <Plus className="h-4 w-4" />
              Add Sign
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="p-6 border-b bg-gray-50">
                     <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
             <div>
               <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
               <input
                 type="text"
                 value={searchTerm}
                 onChange={(e) => setSearchTerm(e.target.value)}
                 placeholder="Search signs..."
                 className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
               />
             </div>
             
             <div>
               <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
               <select
                 value={statusFilter}
                 onChange={(e) => setStatusFilter(e.target.value)}
                 className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
               >
                 <option value="all">All Status</option>
                 <option value="pending">Pending</option>
                 <option value="approved">Approved</option>
                 <option value="rejected">Rejected</option>
               </select>
             </div>
             
             <div>
               <label className="block text-sm font-medium text-gray-700 mb-1">Sign Type</label>
               <select
                 value={signTypeFilter}
                 onChange={(e) => setSignTypeFilter(e.target.value)}
                 className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
               >
                 <option value="all">All Sign Types</option>
                 {signTypes.map((type) => (
                   <option key={type.value} value={type.value}>
                     {type.label}
                   </option>
                 ))}
               </select>
             </div>
             
             <div>
               <label className="block text-sm font-medium text-gray-700 mb-1">Handshape</label>
               <select
                 value={handshapeFilter}
                 onChange={(e) => setHandshapeFilter(e.target.value)}
                 className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
               >
                 <option value="all">All Handshapes</option>
                 <option value="A">A</option>
                 <option value="B">B</option>
                 <option value="C">C</option>
                 <option value="D">D</option>
                 <option value="E">E</option>
                 <option value="F">F</option>
                 <option value="G">G</option>
                 <option value="H">H</option>
                 <option value="I">I</option>
                 <option value="L">L</option>
                 <option value="M">M</option>
                 <option value="N">N</option>
                 <option value="O">O</option>
                 <option value="P">P</option>
                 <option value="Q">Q</option>
                 <option value="R">R</option>
                 <option value="S">S</option>
                 <option value="T">T</option>
                 <option value="U">U</option>
                 <option value="V">V</option>
                 <option value="W">W</option>
                 <option value="X">X</option>
                 <option value="Y">Y</option>
                 <option value="Z">Z</option>
               </select>
             </div>
             
             <div>
               <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
               <select
                 value={locationFilter}
                 onChange={(e) => setLocationFilter(e.target.value)}
                 className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
               >
                 <option value="all">All Locations</option>
                 <option value="neutral space">Neutral Space</option>
                 <option value="head">Head</option>
                 <option value="face">Face</option>
                 <option value="chest">Chest</option>
                 <option value="waist">Waist</option>
                 <option value="chin">Chin</option>
                 <option value="forehead">Forehead</option>
                 <option value="nose">Nose</option>
                 <option value="mouth">Mouth</option>
                 <option value="ear">Ear</option>
                 <option value="eye">Eye</option>
                 <option value="cheek">Cheek</option>
                 <option value="shoulder">Shoulder</option>
                 <option value="arm">Arm</option>
                 <option value="hand">Hand</option>
                 <option value="leg">Leg</option>
                 <option value="foot">Foot</option>
               </select>
             </div>
           </div>
          
          <div className="mt-4 flex gap-2">
            <button
              onClick={fetchSigns}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              <Search className="h-4 w-4" />
              Search
            </button>
            <button
              onClick={() => {
                setSearchTerm('');
                setStatusFilter('all');
                setHandshapeFilter('all');
                setLocationFilter('all');
                setSignTypeFilter('all');
                fetchSigns();
              }}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              <RefreshCw className="h-4 w-4" />
              Clear Filters
            </button>
            
            <button
              onClick={() => {
                if (selectedSigns.size === signs.length) {
                  setSelectedSigns(new Set());
                } else {
                  setSelectedSigns(new Set(signs.map(s => s.id)));
                }
              }}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-700 bg-blue-50 border border-blue-300 rounded-lg hover:bg-blue-100"
            >
              <CheckSquare className="h-4 w-4" />
              {selectedSigns.size === signs.length ? 'Deselect All' : 'Select All'}
            </button>
          </div>
        </div>

        {/* Signs List */}
        <div className="p-6">
          {loading ? (
            <div className="text-center py-8">
              <RefreshCw className="h-8 w-8 text-gray-400 animate-spin mx-auto mb-2" />
              <p className="text-gray-500">Loading signs...</p>
            </div>
          ) : signs.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="h-8 w-8 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-500">No signs found</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {signs.map((sign) => (
                <div key={sign.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start gap-2">
                      <input
                        type="checkbox"
                        checked={selectedSigns.has(sign.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedSigns(prev => new Set([...prev, sign.id]));
                          } else {
                            setSelectedSigns(prev => {
                              const newSet = new Set(prev);
                              newSet.delete(sign.id);
                              return newSet;
                            });
                          }
                        }}
                        className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <div>
                        <h3 className="font-semibold text-gray-900">{sign.gloss}</h3>
                        <p className="text-sm text-gray-600">{sign.english}</p>
                      </div>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      sign.status === 'approved' ? 'bg-green-100 text-green-800' :
                      sign.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {sign.status}
                    </span>
                  </div>
                  
                                     <div className="space-y-2 text-sm">
                     <div className="flex items-center gap-2">
                       <Hash className="h-4 w-4 text-gray-400" />
                       <span className="text-gray-600">Handshape: {sign.handshape}</span>
                     </div>
                     <div className="flex items-center gap-2">
                       <Tag className="h-4 w-4 text-gray-400" />
                       <span className="text-gray-600">Location: {sign.location}</span>
                     </div>
                     <div className="flex items-center gap-2">
                       <Tag className="h-4 w-4 text-gray-400" />
                       <span className="text-gray-600">Type: {sign.sign_type?.replace('_', ' ') || 'isolated sign'}</span>
                     </div>
                     <div className="flex items-center gap-2">
                       <Calendar className="h-4 w-4 text-gray-400" />
                       <span className="text-gray-600">
                         {new Date(sign.uploaded_at).toLocaleDateString()}
                       </span>
                     </div>
                     {sign.video_url && (
                       <div className="flex items-center gap-2">
                         <Video className="h-4 w-4 text-gray-400" />
                         <span className="text-gray-600">Video available</span>
                       </div>
                     )}
                   </div>
                  
                  <div className="mt-4 flex gap-2">
                    <button
                      onClick={() => setSelectedSign(sign)}
                      className="flex-1 px-3 py-1 text-sm text-blue-600 hover:text-blue-800"
                    >
                      View Details
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Sign Detail Modal */}
      {selectedSign && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Sign Details</h2>
                <button
                  onClick={() => setSelectedSign(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <XCircle className="h-6 w-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Gloss</label>
                  <p className="text-gray-900">{selectedSign.gloss}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">English</label>
                  <p className="text-gray-900">{selectedSign.english}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Handshape</label>
                  <p className="text-gray-900">{selectedSign.handshape}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Location</label>
                  <p className="text-gray-900">{selectedSign.location}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Movement</label>
                  <p className="text-gray-900">{selectedSign.movement}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Palm Orientation</label>
                  <p className="text-gray-900">{selectedSign.palm_orientation}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Frequency</label>
                  <p className="text-gray-900">{selectedSign.frequency}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Age of Acquisition</label>
                  <p className="text-gray-900">{selectedSign.age_of_acquisition}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Iconicity</label>
                  <p className="text-gray-900">{selectedSign.iconicity}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Lexical Class</label>
                  <p className="text-gray-900">{selectedSign.lexical_class}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Status</label>
                  <p className="text-gray-900">{selectedSign.status}</p>
                </div>
                                 <div>
                   <label className="block text-sm font-medium text-gray-700">Sign Type</label>
                   <p className="text-gray-900">{selectedSign.sign_type?.replace('_', ' ') || 'isolated sign'}</p>
                 </div>
                 <div>
                   <label className="block text-sm font-medium text-gray-700">Validation Status</label>
                   <p className="text-gray-900">{selectedSign.validation_status}</p>
                 </div>
               </div>
              
              {selectedSign.tags && selectedSign.tags.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Tags</label>
                  <div className="flex flex-wrap gap-2">
                    {selectedSign.tags.map((tag, index) => (
                      <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {selectedSign.notes && (
                <div>
                  <label className="block text-sm font-medium text-gray-700">Notes</label>
                  <p className="text-gray-900">{selectedSign.notes}</p>
                </div>
              )}
              
              {selectedSign.video_url && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Video</label>
                  <video
                    src={selectedSign.video_url}
                    controls
                    className="w-full rounded-lg"
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Video Preview Modal */}
      {showVideoPreview && videoPreviewUrl && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Video Preview</h2>
                <button
                  onClick={() => {
                    setShowVideoPreview(false);
                    URL.revokeObjectURL(videoPreviewUrl);
                    setVideoPreviewUrl(null);
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <video
                src={videoPreviewUrl}
                controls
                className="w-full rounded-lg"
                autoPlay
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
      )}

      {/* Custom Sign Type Modal */}
      {showCustomSignTypeForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Add Custom Sign Type</h2>
                <button
                  onClick={() => setShowCustomSignTypeForm(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Sign Type Name *
                </label>
                <input
                  type="text"
                  value={customSignTypeData.custom_type}
                  onChange={(e) => setCustomSignTypeData(prev => ({ ...prev, custom_type: e.target.value }))}
                  placeholder="e.g., classifier_construction"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Use lowercase with underscores (e.g., classifier_construction)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={customSignTypeData.description}
                  onChange={(e) => setCustomSignTypeData(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe this sign type..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div className="flex gap-2 pt-4">
                <button
                  onClick={addCustomSignType}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Add Sign Type
                </button>
                <button
                  onClick={() => setShowCustomSignTypeForm(false)}
                  className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Batch Update Modal */}
      {showBatchUpdate && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Batch Update Sign Types</h2>
                <button
                  onClick={() => setShowBatchUpdate(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6 space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-blue-800">
                  Update sign type for <strong>{selectedSigns.size}</strong> selected signs
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  New Sign Type *
                </label>
                <select
                  value={batchSignType}
                  onChange={(e) => setBatchSignType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  {signTypes.map((type) => (
                    <option key={type.value} value={type.value}>
                      {type.label} - {type.description}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex gap-2 pt-4">
                <button
                  onClick={batchUpdateSignTypes}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Update Signs
                </button>
                <button
                  onClick={() => setShowBatchUpdate(false)}
                  className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ASL Validation Modal */}
      {showValidation && validationResults && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">ASL Validation Results</h2>
                <button
                  onClick={() => setShowValidation(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6 space-y-4">
              <div className={`p-4 rounded-lg ${
                validationResults.is_valid ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {validationResults.is_valid ? (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-600" />
                  )}
                  <span className={`font-medium ${
                    validationResults.is_valid ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {validationResults.is_valid ? 'Validation Passed' : 'Validation Failed'}
                  </span>
                </div>
                <p className="text-sm text-gray-600">
                  ASL-specific validation completed with {validationResults.errors.length} errors, 
                  {validationResults.warnings.length} warnings, and {validationResults.suggestions.length} suggestions.
                </p>
              </div>

              {validationResults.errors.length > 0 && (
                <div>
                  <h3 className="font-medium text-red-800 mb-2 flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4" />
                    Errors
                  </h3>
                  <ul className="space-y-1">
                    {validationResults.errors.map((error, index) => (
                      <li key={index} className="text-sm text-red-700 bg-red-50 p-2 rounded">
                        {error}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {validationResults.warnings.length > 0 && (
                <div>
                  <h3 className="font-medium text-yellow-800 mb-2 flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4" />
                    Warnings
                  </h3>
                  <ul className="space-y-1">
                    {validationResults.warnings.map((warning, index) => (
                      <li key={index} className="text-sm text-yellow-700 bg-yellow-50 p-2 rounded">
                        {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {validationResults.suggestions.length > 0 && (
                <div>
                  <h3 className="font-medium text-blue-800 mb-2 flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Suggestions
                  </h3>
                  <ul className="space-y-1">
                    {validationResults.suggestions.map((suggestion, index) => (
                      <li key={index} className="text-sm text-blue-700 bg-blue-50 p-2 rounded">
                        {suggestion}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Analytics Modal */}
      {showAnalytics && signTypeAnalytics && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Sign Type Analytics</h2>
                <button
                  onClick={() => setShowAnalytics(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>
            
            <div className="p-6 space-y-6">
              {/* Summary Stats */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-medium text-blue-900">Total Signs</h3>
                  <p className="text-2xl font-bold text-blue-600">{signTypeAnalytics.total_signs}</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <h3 className="font-medium text-green-900">Sign Types</h3>
                  <p className="text-2xl font-bold text-green-600">{Object.keys(signTypeAnalytics.sign_type_counts).length}</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h3 className="font-medium text-purple-900">Custom Types</h3>
                  <p className="text-2xl font-bold text-purple-600">{signTypeAnalytics.custom_types?.length || 0}</p>
                </div>
              </div>

              {/* Top Sign Types */}
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Top Sign Types</h3>
                <div className="space-y-2">
                  {signTypeAnalytics.top_sign_types.map(([type, count]) => (
                    <div key={type} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <span className="font-medium text-gray-900">{type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                        <span className="text-sm text-gray-500 ml-2">({signTypeAnalytics.sign_type_percentages[type]}%)</span>
                      </div>
                      <span className="font-bold text-gray-900">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Sign Type Details */}
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Sign Type Details</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(signTypeAnalytics.sign_type_details).map(([type, details]) => (
                    <div key={type} className="border rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-2">
                        {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span>Count:</span>
                          <span className="font-medium">{details.count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Avg Confidence:</span>
                          <span className="font-medium">{(details.avg_confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Validated:</span>
                          <span className="font-medium">{details.validation_status.validated}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Needs Review:</span>
                          <span className="font-medium">{details.validation_status.needs_review}</span>
                        </div>
                      </div>
                      
                      {details.examples.length > 0 && (
                        <div className="mt-3">
                          <p className="text-xs text-gray-500 mb-1">Examples:</p>
                          <div className="space-y-1">
                            {details.examples.map((example, index) => (
                              <div key={index} className="text-xs bg-gray-100 p-1 rounded">
                                {example.gloss} → {example.english}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Recent Trends */}
              {Object.keys(signTypeAnalytics.recent_trends).length > 0 && (
                <div>
                  <h3 className="font-medium text-gray-900 mb-3">Recent Trends (Last 30 Days)</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {Object.entries(signTypeAnalytics.recent_trends).map(([type, count]) => (
                      <div key={type} className="bg-gray-50 p-3 rounded-lg text-center">
                        <p className="text-sm font-medium text-gray-900">{type.replace('_', ' ')}</p>
                        <p className="text-lg font-bold text-blue-600">{count}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ASLLexDataManager; 