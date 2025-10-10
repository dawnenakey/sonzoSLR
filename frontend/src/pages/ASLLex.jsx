import React, { useState } from 'react';
import { BookOpen, FileSpreadsheet, Upload, Download, Database, Settings, Users, BarChart3, RefreshCw, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/components/ui/use-toast';
import { googleSheetsClient } from '../api/googleSheetsClient';

export default function ASLLex() {
  const [googleSheetsUrl, setGoogleSheetsUrl] = useState('');
  const [syncStatus, setSyncStatus] = useState('idle');
  const [lastSync, setLastSync] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const { toast } = useToast();

  const handleGoogleSheetsSync = async () => {
    if (!googleSheetsUrl) {
      toast({
        variant: "destructive",
        title: "Missing Google Sheets URL",
        description: "Please enter a valid Google Sheets URL to sync with."
      });
      return;
    }

    // Validate URL format
    if (!googleSheetsClient.validateSheetUrl(googleSheetsUrl)) {
      toast({
        variant: "destructive",
        title: "Invalid Google Sheets URL",
        description: "Please enter a valid Google Sheets URL in the format: https://docs.google.com/spreadsheets/d/..."
      });
      return;
    }

    setSyncStatus('syncing');
    try {
      // Use the Google Sheets client to sync data
      const result = await googleSheetsClient.syncFromGoogleSheets(googleSheetsUrl);
      
      setLastSync(new Date().toISOString());
      setSyncStatus('completed');
      
      toast({
        title: "Sync Completed",
        description: `Successfully synced ${result.successful} signs from Google Sheets. ${result.failed > 0 ? `${result.failed} failed.` : ''}`
      });
    } catch (error) {
      setSyncStatus('error');
      toast({
        variant: "destructive",
        title: "Sync Failed",
        description: error.message || "Failed to sync with Google Sheets. Please check the URL and try again."
      });
    }
  };

  const extractSheetId = (url) => {
    const match = url.match(/\/spreadsheets\/d\/([a-zA-Z0-9-_]+)/);
    return match ? match[1] : null;
  };

  const openGoogleSheets = () => {
    if (googleSheetsUrl) {
      window.open(googleSheetsUrl, '_blank');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <BookOpen className="h-8 w-8 text-indigo-600" />
                ASL-LEX Data Management
              </h1>
              <p className="text-gray-600 mt-2">
                Manage ASL sign data and sync with Google Sheets for collaborative data collection
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center gap-2"
              >
                <Settings className="h-4 w-4" />
                Settings
              </Button>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-blue-100 p-2 rounded-lg">
                    <Database className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Total Signs</p>
                    <p className="text-2xl font-bold text-gray-900">Loading...</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-green-100 p-2 rounded-lg">
                    <FileSpreadsheet className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Google Sheets</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {googleSheetsUrl ? 'Connected' : 'Not Connected'}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-purple-100 p-2 rounded-lg">
                    <Users className="h-5 w-5 text-purple-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Data Analysts</p>
                    <p className="text-2xl font-bold text-gray-900">Active</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="bg-amber-100 p-2 rounded-lg">
                    <BarChart3 className="h-5 w-5 text-amber-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Validation Rate</p>
                    <p className="text-2xl font-bold text-gray-900">Loading...</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Google Sheets Integration Section */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileSpreadsheet className="h-5 w-5" />
              Google Sheets Integration
            </CardTitle>
            <CardDescription>
              Connect your Google Sheets to automatically sync ASL sign data with the database
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Google Sheets URL
                  </label>
                  <Input
                    type="url"
                    placeholder="https://docs.google.com/spreadsheets/d/..."
                    value={googleSheetsUrl}
                    onChange={(e) => setGoogleSheetsUrl(e.target.value)}
                    className="w-full"
                  />
                </div>
                <Button
                  onClick={openGoogleSheets}
                  variant="outline"
                  disabled={!googleSheetsUrl}
                  className="flex items-center gap-2"
                >
                  <ExternalLink className="h-4 w-4" />
                  Open Sheet
                </Button>
              </div>
              
              <div className="flex items-center gap-4">
                <Button
                  onClick={handleGoogleSheetsSync}
                  disabled={!googleSheetsUrl || syncStatus === 'syncing'}
                  className="flex items-center gap-2"
                >
                  {syncStatus === 'syncing' ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Upload className="h-4 w-4" />
                  )}
                  {syncStatus === 'syncing' ? 'Syncing...' : 'Sync from Google Sheets'}
                </Button>
                
                {lastSync && (
                  <div className="text-sm text-gray-600">
                    Last synced: {new Date(lastSync).toLocaleString()}
                  </div>
                )}
                
                {syncStatus === 'completed' && (
                  <Badge variant="default" className="bg-green-100 text-green-800">
                    Synced Successfully
                  </Badge>
                )}
                
                {syncStatus === 'error' && (
                  <Badge variant="destructive">
                    Sync Failed
                  </Badge>
                )}
              </div>
              
              <Alert>
                <AlertTitle>Google Sheets Format Requirements</AlertTitle>
                <AlertDescription>
                  Your Google Sheet should have the following columns: Gloss, English, Handshape, Location, Movement, 
                  Palm Orientation, Dominant Hand, Non-Dominant Hand, Frequency, Age of Acquisition, Iconicity, 
                  Lexical Class, Tags, Notes. The first row should contain headers.
                </AlertDescription>
              </Alert>
            </div>
          </CardContent>
        </Card>

        {/* ASL-LEX Data Manager */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              ASL-LEX Database Management
            </CardTitle>
            <CardDescription>
              Manage individual signs, validate data, and export to various formats
            </CardDescription>
          </CardHeader>
          <CardContent>
    
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 