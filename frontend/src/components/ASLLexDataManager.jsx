import React, { useState, useEffect } from 'react';
import { 
  Upload, Database, Search, Filter, Download, Trash2, Edit2, Eye, 
  Plus, FileText, BarChart3, CheckCircle, AlertCircle, Loader2,
  Calendar, Clock, Hash, Tag, Users, Globe, BookOpen, Zap
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

// Mock ASL-LEX data structure
const ASL_LEX_SIGN_TEMPLATE = {
  id: '',
  gloss: '', // The sign gloss (e.g., "HELLO")
  english: '', // English translation
  handshape: '', // Handshape classification
  location: '', // Location on body
  movement: '', // Movement type
  palm_orientation: '', // Palm orientation
  dominant_hand: '', // Dominant hand shape
  non_dominant_hand: '', // Non-dominant hand shape
  video_url: '', // URL to the sign video
  frequency: 0, // Frequency in ASL-LEX corpus
  age_of_acquisition: 0, // Age of acquisition
  iconicity: 0, // Iconicity rating
  lexical_class: '', // Lexical class (noun, verb, etc.)
  tags: [], // Additional tags
  notes: '', // Additional notes
  uploaded_by: '', // Who uploaded this sign
  uploaded_at: '', // When it was uploaded
  status: 'pending', // pending, approved, rejected
  confidence_score: 0, // AI confidence score
  validation_status: 'unvalidated' // unvalidated, validated, needs_review
};

export default function ASLLexDataManager() {
  const [signs, setSigns] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterHandshape, setFilterHandshape] = useState('all');
  const [filterLocation, setFilterLocation] = useState('all');
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [selectedSign, setSelectedSign] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [stats, setStats] = useState({
    total: 0,
    approved: 0,
    pending: 0,
    rejected: 0,
    validated: 0,
    unvalidated: 0
  });
  
  const { toast } = useToast();

  // Mock data for demonstration
  const mockSigns = [
    {
      id: 'asl-001',
      gloss: 'HELLO',
      english: 'Hello',
      handshape: 'B',
      location: 'neutral space',
      movement: 'wave',
      palm_orientation: 'palm forward',
      dominant_hand: 'B',
      non_dominant_hand: 'B',
      video_url: 'https://asl-lex.org/videos/hello.mp4',
      frequency: 95,
      age_of_acquisition: 2.5,
      iconicity: 0.8,
      lexical_class: 'interjection',
      tags: ['greeting', 'common', 'basic'],
      notes: 'One of the most common signs in ASL',
      uploaded_by: 'data_analyst_1',
      uploaded_at: '2024-01-15T10:30:00Z',
      status: 'approved',
      confidence_score: 0.95,
      validation_status: 'validated'
    },
    {
      id: 'asl-002',
      gloss: 'THANK-YOU',
      english: 'Thank you',
      handshape: 'A',
      location: 'chin',
      movement: 'forward',
      palm_orientation: 'palm up',
      dominant_hand: 'A',
      non_dominant_hand: 'B',
      video_url: 'https://asl-lex.org/videos/thank_you.mp4',
      frequency: 87,
      age_of_acquisition: 3.2,
      iconicity: 0.6,
      lexical_class: 'interjection',
      tags: ['politeness', 'common'],
      notes: 'Palm orientation varies by region',
      uploaded_by: 'data_analyst_2',
      uploaded_at: '2024-01-16T14:20:00Z',
      status: 'pending',
      confidence_score: 0.78,
      validation_status: 'needs_review'
    },
    {
      id: 'asl-003',
      gloss: 'GOOD',
      english: 'Good',
      handshape: 'B',
      location: 'chin',
      movement: 'downward',
      palm_orientation: 'palm down',
      dominant_hand: 'B',
      non_dominant_hand: 'B',
      video_url: 'https://asl-lex.org/videos/good.mp4',
      frequency: 92,
      age_of_acquisition: 2.8,
      iconicity: 0.7,
      lexical_class: 'adjective',
      tags: ['evaluation', 'positive'],
      notes: 'Often used in compound signs',
      uploaded_by: 'data_analyst_1',
      uploaded_at: '2024-01-17T09:15:00Z',
      status: 'approved',
      confidence_score: 0.89,
      validation_status: 'validated'
    }
  ];

  useEffect(() => {
    loadSigns();
    calculateStats();
  }, []);

  const loadSigns = async () => {
    setIsLoading(true);
    try {
      // In a real implementation, this would call the AWS API
      // const response = await fetch('/api/asl-lex/signs');
      // const data = await response.json();
      
      // For now, use mock data
      await new Promise(resolve => setTimeout(resolve, 1000));
      setSigns(mockSigns);
    } catch (error) {
      console.error('Error loading ASL-LEX signs:', error);
      toast({
        variant: "destructive",
        title: "Error Loading Signs",
        description: "Could not load ASL-LEX signs from database"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const calculateStats = () => {
    const stats = {
      total: signs.length,
      approved: signs.filter(s => s.status === 'approved').length,
      pending: signs.filter(s => s.status === 'pending').length,
      rejected: signs.filter(s => s.status === 'rejected').length,
      validated: signs.filter(s => s.validation_status === 'validated').length,
      unvalidated: signs.filter(s => s.validation_status === 'unvalidated').length
    };
    setStats(stats);
  };

  const handleUploadSign = async (signData) => {
    setUploading(true);
    setUploadProgress(0);
    
    try {
      // Simulate upload progress
      for (let i = 0; i <= 100; i += 10) {
        setUploadProgress(i);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      // In a real implementation, this would upload to AWS S3
      const newSign = {
        ...signData,
        id: `asl-${Date.now()}`,
        uploaded_at: new Date().toISOString(),
        status: 'pending',
        validation_status: 'unvalidated',
        confidence_score: Math.random() * 0.3 + 0.7 // Mock confidence score
      };
      
      setSigns(prev => [newSign, ...prev]);
      setShowUploadForm(false);
      
      toast({
        title: "Sign Uploaded Successfully",
        description: `"${signData.gloss}" has been added to the ASL-LEX database`
      });
      
    } catch (error) {
      console.error('Error uploading sign:', error);
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: "Could not upload sign to database"
      });
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleDeleteSign = async (signId) => {
    try {
      // In a real implementation, this would call the AWS API
      setSigns(prev => prev.filter(s => s.id !== signId));
      
      toast({
        title: "Sign Deleted",
        description: "Sign has been removed from the database"
      });
    } catch (error) {
      console.error('Error deleting sign:', error);
      toast({
        variant: "destructive",
        title: "Delete Failed",
        description: "Could not delete sign from database"
      });
    }
  };

  const handleUpdateSign = async (signId, updates) => {
    try {
      setSigns(prev => prev.map(s => 
        s.id === signId ? { ...s, ...updates } : s
      ));
      
      toast({
        title: "Sign Updated",
        description: "Sign information has been updated"
      });
    } catch (error) {
      console.error('Error updating sign:', error);
      toast({
        variant: "destructive",
        title: "Update Failed",
        description: "Could not update sign information"
      });
    }
  };

  const filteredSigns = signs.filter(sign => {
    const matchesSearch = sign.gloss.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         sign.english.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || sign.status === filterStatus;
    const matchesHandshape = filterHandshape === 'all' || sign.handshape === filterHandshape;
    const matchesLocation = filterLocation === 'all' || sign.location === filterLocation;
    
    return matchesSearch && matchesStatus && matchesHandshape && matchesLocation;
  });

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved': return 'bg-green-100 text-green-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'rejected': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getValidationColor = (status) => {
    switch (status) {
      case 'validated': return 'bg-blue-100 text-blue-800';
      case 'needs_review': return 'bg-orange-100 text-orange-800';
      case 'unvalidated': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">ASL-LEX Data Manager</h2>
          <p className="text-gray-600">Manage ASL-LEX signs in the AWS S3 database</p>
        </div>
        <Button onClick={() => setShowUploadForm(true)} className="flex items-center gap-2">
          <Plus className="h-4 w-4" />
          Add New Sign
        </Button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Database className="h-5 w-5 text-blue-600" />
              <div>
                <p className="text-sm text-gray-600">Total Signs</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <div>
                <p className="text-sm text-gray-600">Approved</p>
                <p className="text-2xl font-bold text-gray-900">{stats.approved}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-yellow-600" />
              <div>
                <p className="text-sm text-gray-600">Pending</p>
                <p className="text-2xl font-bold text-gray-900">{stats.pending}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-5 w-5 text-purple-600" />
              <div>
                <p className="text-sm text-gray-600">Validated</p>
                <p className="text-2xl font-bold text-gray-900">{stats.validated}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Eye className="h-5 w-5 text-indigo-600" />
              <div>
                <p className="text-sm text-gray-600">Needs Review</p>
                <p className="text-2xl font-bold text-gray-900">{stats.unvalidated}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Zap className="h-5 w-5 text-amber-600" />
              <div>
                <p className="text-sm text-gray-600">Avg Confidence</p>
                <p className="text-2xl font-bold text-gray-900">
                  {signs.length > 0 ? Math.round(signs.reduce((sum, s) => sum + s.confidence_score, 0) / signs.length * 100) : 0}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Search & Filter
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
              <Input
                placeholder="Search by gloss or English..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="approved">Approved</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="rejected">Rejected</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Handshape</label>
              <Select value={filterHandshape} onValueChange={setFilterHandshape}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Handshapes</SelectItem>
                  <SelectItem value="A">A</SelectItem>
                  <SelectItem value="B">B</SelectItem>
                  <SelectItem value="C">C</SelectItem>
                  <SelectItem value="D">D</SelectItem>
                  <SelectItem value="E">E</SelectItem>
                  <SelectItem value="F">F</SelectItem>
                  <SelectItem value="G">G</SelectItem>
                  <SelectItem value="H">H</SelectItem>
                  <SelectItem value="I">I</SelectItem>
                  <SelectItem value="L">L</SelectItem>
                  <SelectItem value="M">M</SelectItem>
                  <SelectItem value="N">N</SelectItem>
                  <SelectItem value="O">O</SelectItem>
                  <SelectItem value="R">R</SelectItem>
                  <SelectItem value="S">S</SelectItem>
                  <SelectItem value="T">T</SelectItem>
                  <SelectItem value="U">U</SelectItem>
                  <SelectItem value="V">V</SelectItem>
                  <SelectItem value="W">W</SelectItem>
                  <SelectItem value="X">X</SelectItem>
                  <SelectItem value="Y">Y</SelectItem>
                  <SelectItem value="Z">Z</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
              <Select value={filterLocation} onValueChange={setFilterLocation}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Locations</SelectItem>
                  <SelectItem value="neutral space">Neutral Space</SelectItem>
                  <SelectItem value="chin">Chin</SelectItem>
                  <SelectItem value="forehead">Forehead</SelectItem>
                  <SelectItem value="chest">Chest</SelectItem>
                  <SelectItem value="shoulder">Shoulder</SelectItem>
                  <SelectItem value="ear">Ear</SelectItem>
                  <SelectItem value="nose">Nose</SelectItem>
                  <SelectItem value="mouth">Mouth</SelectItem>
                  <SelectItem value="eye">Eye</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Upload Form */}
      {showUploadForm && (
        <Card>
          <CardHeader>
            <CardTitle>Add New ASL-LEX Sign</CardTitle>
            <CardDescription>Upload a new sign to the ASL-LEX database</CardDescription>
          </CardHeader>
          <CardContent>
            <SignUploadForm 
              onSubmit={handleUploadSign}
              onCancel={() => setShowUploadForm(false)}
              uploading={uploading}
              progress={uploadProgress}
            />
          </CardContent>
        </Card>
      )}

      {/* Signs List */}
      <Card>
        <CardHeader>
          <CardTitle>ASL-LEX Signs ({filteredSigns.length})</CardTitle>
          <CardDescription>Manage and review uploaded signs</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
              <span className="ml-2 text-gray-600">Loading signs...</span>
            </div>
          ) : filteredSigns.length === 0 ? (
            <div className="text-center py-8">
              <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No signs found</h3>
              <p className="text-gray-600">Try adjusting your search or filters</p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredSigns.map((sign) => (
                <SignCard
                  key={sign.id}
                  sign={sign}
                  onEdit={setSelectedSign}
                  onDelete={handleDeleteSign}
                  onUpdate={handleUpdateSign}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sign Detail Modal */}
      {selectedSign && (
        <SignDetailModal
          sign={selectedSign}
          onClose={() => setSelectedSign(null)}
          onUpdate={handleUpdateSign}
        />
      )}
    </div>
  );
}

// Sign Upload Form Component
function SignUploadForm({ onSubmit, onCancel, uploading, progress }) {
  const [formData, setFormData] = useState({
    gloss: '',
    english: '',
    handshape: '',
    location: '',
    movement: '',
    palm_orientation: '',
    dominant_hand: '',
    non_dominant_hand: '',
    video_url: '',
    frequency: 0,
    age_of_acquisition: 0,
    iconicity: 0,
    lexical_class: '',
    tags: '',
    notes: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    const signData = {
      ...formData,
      tags: formData.tags.split(',').map(tag => tag.trim()).filter(tag => tag)
    };
    onSubmit(signData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Gloss *</label>
          <Input
            required
            value={formData.gloss}
            onChange={(e) => setFormData(prev => ({ ...prev, gloss: e.target.value }))}
            placeholder="e.g., HELLO"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">English Translation *</label>
          <Input
            required
            value={formData.english}
            onChange={(e) => setFormData(prev => ({ ...prev, english: e.target.value }))}
            placeholder="e.g., Hello"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Handshape</label>
          <Select value={formData.handshape} onValueChange={(value) => setFormData(prev => ({ ...prev, handshape: value }))}>
            <SelectTrigger>
              <SelectValue placeholder="Select handshape" />
            </SelectTrigger>
            <SelectContent>
              {['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'].map(h => (
                <SelectItem key={h} value={h}>{h}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
          <Select value={formData.location} onValueChange={(value) => setFormData(prev => ({ ...prev, location: value }))}>
            <SelectTrigger>
              <SelectValue placeholder="Select location" />
            </SelectTrigger>
            <SelectContent>
              {['neutral space', 'chin', 'forehead', 'chest', 'shoulder', 'ear', 'nose', 'mouth', 'eye'].map(l => (
                <SelectItem key={l} value={l}>{l}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Movement</label>
          <Input
            value={formData.movement}
            onChange={(e) => setFormData(prev => ({ ...prev, movement: e.target.value }))}
            placeholder="e.g., wave, forward, downward"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Palm Orientation</label>
          <Input
            value={formData.palm_orientation}
            onChange={(e) => setFormData(prev => ({ ...prev, palm_orientation: e.target.value }))}
            placeholder="e.g., palm forward, palm up"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Video URL</label>
          <Input
            value={formData.video_url}
            onChange={(e) => setFormData(prev => ({ ...prev, video_url: e.target.value }))}
            placeholder="https://asl-lex.org/videos/sign.mp4"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Lexical Class</label>
          <Select value={formData.lexical_class} onValueChange={(value) => setFormData(prev => ({ ...prev, lexical_class: value }))}>
            <SelectTrigger>
              <SelectValue placeholder="Select class" />
            </SelectTrigger>
            <SelectContent>
              {['noun', 'verb', 'adjective', 'adverb', 'interjection', 'pronoun', 'conjunction', 'preposition'].map(c => (
                <SelectItem key={c} value={c}>{c}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Frequency</label>
          <Input
            type="number"
            value={formData.frequency}
            onChange={(e) => setFormData(prev => ({ ...prev, frequency: parseInt(e.target.value) || 0 }))}
            placeholder="0-100"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Age of Acquisition</label>
          <Input
            type="number"
            step="0.1"
            value={formData.age_of_acquisition}
            onChange={(e) => setFormData(prev => ({ ...prev, age_of_acquisition: parseFloat(e.target.value) || 0 }))}
            placeholder="e.g., 2.5"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Iconicity (0-1)</label>
          <Input
            type="number"
            step="0.1"
            min="0"
            max="1"
            value={formData.iconicity}
            onChange={(e) => setFormData(prev => ({ ...prev, iconicity: parseFloat(e.target.value) || 0 }))}
            placeholder="0.0-1.0"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Tags</label>
          <Input
            value={formData.tags}
            onChange={(e) => setFormData(prev => ({ ...prev, tags: e.target.value }))}
            placeholder="comma-separated tags"
          />
        </div>
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
        <Textarea
          value={formData.notes}
          onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
          placeholder="Additional notes about this sign..."
          rows={3}
        />
      </div>

      {uploading && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Uploading to AWS S3...</span>
            <span>{progress}%</span>
          </div>
          <Progress value={progress} />
        </div>
      )}

      <div className="flex gap-3">
        <Button type="submit" disabled={uploading}>
          {uploading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Uploading...
            </>
          ) : (
            <>
              <Upload className="h-4 w-4 mr-2" />
              Upload Sign
            </>
          )}
        </Button>
        <Button type="button" variant="outline" onClick={onCancel} disabled={uploading}>
          Cancel
        </Button>
      </div>
    </form>
  );
}

// Sign Card Component
function SignCard({ sign, onEdit, onDelete, onUpdate }) {
  const [showActions, setShowActions] = useState(false);

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-lg font-semibold text-gray-900">{sign.gloss}</h3>
              <Badge className={getStatusColor(sign.status)}>{sign.status}</Badge>
              <Badge className={getValidationColor(sign.validation_status)}>{sign.validation_status}</Badge>
            </div>
            
            <p className="text-gray-600 mb-2">{sign.english}</p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div>
                <span className="text-gray-500">Handshape:</span>
                <span className="ml-1 font-medium">{sign.handshape}</span>
              </div>
              <div>
                <span className="text-gray-500">Location:</span>
                <span className="ml-1 font-medium">{sign.location}</span>
              </div>
              <div>
                <span className="text-gray-500">Frequency:</span>
                <span className="ml-1 font-medium">{sign.frequency}</span>
              </div>
              <div>
                <span className="text-gray-500">Confidence:</span>
                <span className="ml-1 font-medium">{Math.round(sign.confidence_score * 100)}%</span>
              </div>
            </div>
            
            <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
              <Calendar className="h-3 w-3" />
              <span>{new Date(sign.uploaded_at).toLocaleDateString()}</span>
              <Users className="h-3 w-3" />
              <span>{sign.uploaded_by}</span>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onEdit(sign)}
            >
              <Eye className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDelete(sign.id)}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Sign Detail Modal Component
function SignDetailModal({ sign, onClose, onUpdate }) {
  const [editing, setEditing] = useState(false);
  const [formData, setFormData] = useState(sign);

  const handleSave = () => {
    onUpdate(sign.id, formData);
    setEditing(false);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">{sign.gloss}</h2>
          <div className="flex gap-2">
            {editing ? (
              <>
                <Button size="sm" onClick={handleSave}>Save</Button>
                <Button size="sm" variant="outline" onClick={() => setEditing(false)}>Cancel</Button>
              </>
            ) : (
              <>
                <Button size="sm" onClick={() => setEditing(true)}>
                  <Edit2 className="h-4 w-4 mr-2" />
                  Edit
                </Button>
                <Button size="sm" variant="outline" onClick={onClose}>Close</Button>
              </>
            )}
          </div>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">English</label>
              {editing ? (
                <Input
                  value={formData.english}
                  onChange={(e) => setFormData(prev => ({ ...prev, english: e.target.value }))}
                />
              ) : (
                <p className="text-gray-900">{sign.english}</p>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">Handshape</label>
              {editing ? (
                <Input
                  value={formData.handshape}
                  onChange={(e) => setFormData(prev => ({ ...prev, handshape: e.target.value }))}
                />
              ) : (
                <p className="text-gray-900">{sign.handshape}</p>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">Location</label>
              {editing ? (
                <Input
                  value={formData.location}
                  onChange={(e) => setFormData(prev => ({ ...prev, location: e.target.value }))}
                />
              ) : (
                <p className="text-gray-900">{sign.location}</p>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">Movement</label>
              {editing ? (
                <Input
                  value={formData.movement}
                  onChange={(e) => setFormData(prev => ({ ...prev, movement: e.target.value }))}
                />
              ) : (
                <p className="text-gray-900">{sign.movement}</p>
              )}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Notes</label>
            {editing ? (
              <Textarea
                value={formData.notes}
                onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
                rows={3}
              />
            ) : (
              <p className="text-gray-900">{sign.notes}</p>
            )}
          </div>
          
          <div className="flex gap-2">
            <Badge className={getStatusColor(sign.status)}>{sign.status}</Badge>
            <Badge className={getValidationColor(sign.validation_status)}>{sign.validation_status}</Badge>
            <Badge variant="outline">Confidence: {Math.round(sign.confidence_score * 100)}%</Badge>
          </div>
        </div>
      </div>
    </div>
  );
} 