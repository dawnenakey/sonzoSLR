import React, { useState, useEffect } from 'react';
import { 
  BookOpen, 
  Search, 
  Plus, 
  Filter, 
  Download, 
  Upload, 
  Edit, 
  Trash2, 
  Eye, 
  CheckCircle, 
  XCircle, 
  Clock,
  HandMetal,
  MapPin,
  Move,
  RotateCw,
  Users,
  BarChart3,
  Settings,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { 
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useToast } from '@/components/ui/use-toast';

export default function LexiconManagement() {
  const [signs, setSigns] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [handshapeFilter, setHandshapeFilter] = useState('all');
  const [locationFilter, setLocationFilter] = useState('all');
  const [signTypeFilter, setSignTypeFilter] = useState('all');
  const [sortBy, setSortBy] = useState('gloss');
  const [sortOrder, setSortOrder] = useState('asc');
  const [selectedSigns, setSelectedSigns] = useState([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [editingSign, setEditingSign] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const { toast } = useToast();

  // Handshapes A-Z
  const handshapes = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
  
  // Common ASL locations
  const locations = [
    'neutral space', 'chin', 'forehead', 'chest', 'shoulder', 'ear', 'nose', 'mouth',
    'eye', 'cheek', 'neck', 'waist', 'hip', 'thigh', 'knee', 'ankle', 'foot'
  ];

  // Sign types
  const signTypes = [
    'isolated_sign', 'continuous_signing', 'fingerspelling', 'classifier', 
    'compound_sign', 'inflection', 'derivation', 'borrowing'
  ];

  useEffect(() => {
    fetchSigns();
    fetchStatistics();
  }, []);

  const fetchSigns = async () => {
    try {
      setIsLoading(true);
      const params = new URLSearchParams({
        status: statusFilter,
        handshape: handshapeFilter,
        location: locationFilter,
        search: searchQuery
      });

      const response = await fetch(`/api/asl-lex/signs?${params}`);
      if (!response.ok) throw new Error('Failed to fetch signs');
      
      const data = await response.json();
      setSigns(data);
    } catch (error) {
      console.error('Error fetching signs:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load signs from the lexicon"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch('/api/asl-lex/statistics');
      if (!response.ok) throw new Error('Failed to fetch statistics');
      
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    }
  };

  const handleSearch = () => {
    fetchSigns();
  };

  const handleClearFilters = () => {
    setSearchQuery('');
    setStatusFilter('all');
    setHandshapeFilter('all');
    setLocationFilter('all');
    setSignTypeFilter('all');
    fetchSigns();
  };

  const handleSignSelect = (signId) => {
    setSelectedSigns(prev => 
      prev.includes(signId) 
        ? prev.filter(id => id !== signId)
        : [...prev, signId]
    );
  };

  const handleSelectAll = () => {
    if (selectedSigns.length === filteredSigns.length) {
      setSelectedSigns([]);
    } else {
      setSelectedSigns(filteredSigns.map(sign => sign.id));
    }
  };

  const handleDeleteSign = async (signId) => {
    try {
      const response = await fetch(`/api/asl-lex/signs/${signId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) throw new Error('Failed to delete sign');
      
      setSigns(prev => prev.filter(sign => sign.id !== signId));
      toast({
        title: "Sign Deleted",
        description: "The sign has been removed from the lexicon"
      });
    } catch (error) {
      console.error('Error deleting sign:', error);
      toast({
        variant: "destructive",
        title: "Delete Failed",
        description: "Could not delete the sign"
      });
    }
  };

  const handleBatchUpdate = async (updates) => {
    try {
      const response = await fetch('/api/asl-lex/signs/batch-update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sign_ids: selectedSigns,
          updates
        })
      });
      
      if (!response.ok) throw new Error('Failed to update signs');
      
      await fetchSigns();
      setSelectedSigns([]);
      toast({
        title: "Batch Update Complete",
        description: `${selectedSigns.length} signs have been updated`
      });
    } catch (error) {
      console.error('Error updating signs:', error);
      toast({
        variant: "destructive",
        title: "Update Failed",
        description: "Could not update the selected signs"
      });
    }
  };

  const handleExport = async () => {
    try {
      const response = await fetch('/api/asl-lex/export', {
        method: 'GET'
      });
      
      if (!response.ok) throw new Error('Export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `asl_lexicon_export_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast({
        title: "Export Complete",
        description: "Lexicon data exported successfully"
      });
    } catch (error) {
      console.error('Export failed:', error);
      toast({
        variant: "destructive",
        title: "Export Failed",
        description: "Could not export lexicon data"
      });
    }
  };

  const filteredSigns = signs.filter(sign => {
    const matchesSearch = searchQuery === '' || 
      sign.gloss.toLowerCase().includes(searchQuery.toLowerCase()) ||
      sign.english.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || sign.status === statusFilter;
    const matchesHandshape = handshapeFilter === 'all' || sign.handshape === handshapeFilter;
    const matchesLocation = locationFilter === 'all' || sign.location === locationFilter;
    const matchesSignType = signTypeFilter === 'all' || sign.sign_type === signTypeFilter;
    
    return matchesSearch && matchesStatus && matchesHandshape && matchesLocation && matchesSignType;
  }).sort((a, b) => {
    const aValue = a[sortBy] || '';
    const bValue = b[sortBy] || '';
    
    if (sortOrder === 'asc') {
      return aValue.localeCompare(bValue);
    } else {
      return bValue.localeCompare(aValue);
    }
  });

  const getStatusIcon = (status) => {
    switch (status) {
      case 'approved': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'pending': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'rejected': return <XCircle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved': return 'bg-green-100 text-green-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'rejected': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-gray-500" />
          <span className="ml-2 text-gray-500">Loading lexicon...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <BookOpen className="h-8 w-8 text-indigo-600" />
              ASL Lexicon Management
            </h1>
            <p className="text-gray-600 mt-2">
              Manage and organize your American Sign Language lexicon database
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={handleExport} variant="outline" className="flex items-center gap-2">
              <Download className="h-4 w-4" />
              Export
            </Button>
            <Button onClick={() => setShowAddDialog(true)} className="flex items-center gap-2">
              <Plus className="h-4 w-4" />
              Add Sign
            </Button>
          </div>
        </div>

        {/* Statistics Cards */}
        {statistics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-blue-100 p-2 rounded-lg">
                    <BookOpen className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Total Signs</p>
                    <p className="text-2xl font-bold text-gray-900">{statistics.total_signs}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-green-100 p-2 rounded-lg">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Approved</p>
                    <p className="text-2xl font-bold text-gray-900">{statistics.approved_signs}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-yellow-100 p-2 rounded-lg">
                    <Clock className="h-5 w-5 text-yellow-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Pending</p>
                    <p className="text-2xl font-bold text-gray-900">{statistics.pending_signs}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-purple-100 p-2 rounded-lg">
                    <BarChart3 className="h-5 w-5 text-purple-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Validation Rate</p>
                    <p className="text-2xl font-bold text-gray-900">{statistics.validation_rate}%</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Search and Filters */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Search & Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
              <div className="relative">
                <Search className="h-4 w-4 absolute left-2.5 top-1/2 transform -translate-y-1/2 text-gray-500" />
                <Input
                  placeholder="Search gloss or English..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="All Statuses" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value="approved">Approved</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="rejected">Rejected</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Handshape</label>
              <Select value={handshapeFilter} onValueChange={setHandshapeFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="All Handshapes" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Handshapes</SelectItem>
                  {handshapes.map(shape => (
                    <SelectItem key={shape} value={shape}>{shape}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
              <Select value={locationFilter} onValueChange={setLocationFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="All Locations" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Locations</SelectItem>
                  {locations.map(location => (
                    <SelectItem key={location} value={location}>{location}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Sign Type</label>
              <Select value={signTypeFilter} onValueChange={setSignTypeFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="All Types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  {signTypes.map(type => (
                    <SelectItem key={type} value={type}>{type.replace('_', ' ')}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button onClick={handleSearch} className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Search
            </Button>
            <Button onClick={handleClearFilters} variant="outline" className="flex items-center gap-2">
              <XCircle className="h-4 w-4" />
              Clear Filters
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Batch Actions */}
      {selectedSigns.length > 0 && (
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">
                {selectedSigns.length} sign{selectedSigns.length !== 1 ? 's' : ''} selected
              </span>
              <div className="flex items-center gap-2">
                <Button 
                  onClick={() => handleBatchUpdate({ status: 'approved' })}
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2"
                >
                  <CheckCircle className="h-4 w-4" />
                  Approve
                </Button>
                <Button 
                  onClick={() => handleBatchUpdate({ status: 'rejected' })}
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2"
                >
                  <XCircle className="h-4 w-4" />
                  Reject
                </Button>
                <Button 
                  onClick={() => setSelectedSigns([])}
                  variant="outline"
                  size="sm"
                >
                  Clear Selection
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Signs Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Lexicon Entries ({filteredSigns.length})</CardTitle>
            <div className="flex items-center gap-2">
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gloss">Gloss</SelectItem>
                  <SelectItem value="english">English</SelectItem>
                  <SelectItem value="handshape">Handshape</SelectItem>
                  <SelectItem value="location">Location</SelectItem>
                  <SelectItem value="created_at">Date Created</SelectItem>
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              >
                {sortOrder === 'asc' ? '↑' : '↓'}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">
                  <input
                    type="checkbox"
                    checked={selectedSigns.length === filteredSigns.length && filteredSigns.length > 0}
                    onChange={handleSelectAll}
                    className="rounded"
                  />
                </TableHead>
                <TableHead>Gloss</TableHead>
                <TableHead>English</TableHead>
                <TableHead>Handshape</TableHead>
                <TableHead>Location</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredSigns.map((sign) => (
                <TableRow key={sign.id}>
                  <TableCell>
                    <input
                      type="checkbox"
                      checked={selectedSigns.includes(sign.id)}
                      onChange={() => handleSignSelect(sign.id)}
                      className="rounded"
                    />
                  </TableCell>
                  <TableCell className="font-medium">{sign.gloss}</TableCell>
                  <TableCell>{sign.english}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="flex items-center gap-1 w-fit">
                      <HandMetal className="h-3 w-3" />
                      {sign.handshape}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="flex items-center gap-1 w-fit">
                      <MapPin className="h-3 w-3" />
                      {sign.location}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">
                      {sign.sign_type?.replace('_', ' ') || 'N/A'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge className={`flex items-center gap-1 w-fit ${getStatusColor(sign.status)}`}>
                      {getStatusIcon(sign.status)}
                      {sign.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Button variant="ghost" size="sm">
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => handleDeleteSign(sign.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          
          {filteredSigns.length === 0 && (
            <div className="text-center py-8">
              <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No signs found</h3>
              <p className="text-gray-500 mb-4">
                Try adjusting your search criteria or add new signs to the lexicon.
              </p>
              <Button onClick={() => setShowAddDialog(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add First Sign
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add Sign Dialog */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add New Sign to Lexicon</DialogTitle>
            <DialogDescription>
              Add a new ASL sign with comprehensive metadata to the lexicon database.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Gloss *</label>
                <Input placeholder="HELLO" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">English Translation *</label>
                <Input placeholder="Hello" />
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Handshape *</label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select handshape" />
                  </SelectTrigger>
                  <SelectContent>
                    {handshapes.map(shape => (
                      <SelectItem key={shape} value={shape}>{shape}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Location *</label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select location" />
                  </SelectTrigger>
                  <SelectContent>
                    {locations.map(location => (
                      <SelectItem key={location} value={location}>{location}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Sign Type</label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    {signTypes.map(type => (
                      <SelectItem key={type} value={type}>{type.replace('_', ' ')}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                Cancel
              </Button>
              <Button>
                Add Sign
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
