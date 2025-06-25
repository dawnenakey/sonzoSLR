
import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Check, Copy, Download } from 'lucide-react';
import { formatTime } from './timeUtils';

export default function ExportJsonDialog({ 
  isOpen, 
  onClose, 
  annotations, 
  videoTitle 
}) {
  const [copied, setCopied] = useState(false);
  
  // Format annotations for export (match expected JSON output)
  const formatAnnotationsForExport = () => {
    return annotations.map(annotation => ({
      segment_type: annotation.segment_type,
      start_time: formatTime(annotation.start_time),
      duration: annotation.duration.toFixed(2),
      description: annotation.description || ""
    }));
  };
  
  const exportedJson = JSON.stringify(formatAnnotationsForExport(), null, 2);
  
  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(exportedJson);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const handleDownload = () => {
    const blob = new Blob([exportedJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${videoTitle.replace(/\s+/g, '_')}_annotations.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Export Annotations</DialogTitle>
          <DialogDescription>
            Your annotations in JSON format
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>JSON Output</Label>
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-8 px-2 text-xs"
                onClick={handleCopyToClipboard}
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 mr-1" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5 mr-1" />
                    Copy
                  </>
                )}
              </Button>
            </div>
            <div className="relative">
              <pre className="bg-gray-50 p-4 rounded-md text-xs max-h-80 overflow-y-auto overflow-x-auto font-mono">
                {exportedJson}
              </pre>
            </div>
          </div>
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button onClick={handleDownload} className="gap-2">
            <Download className="h-4 w-4" />
            Download JSON
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
