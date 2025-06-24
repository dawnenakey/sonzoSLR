import React, { useState } from 'react';
import { Button } from "./ui/button";
import { PlayCircle, Pause } from 'lucide-react';
import { Input } from "./ui/input";
import { Label } from "./ui/label";

interface AnnotationControlsProps {
  isSegmenting: boolean;
  onSegmentStart: () => void;
  currentSegmentDuration: number;
  onSegmentEnd: (type: string) => void;
  onCancelSegment: () => void;
  onCreateAnnotation: (startTime: number, endTime: number, label: string, notes?: string) => void;
}

export default function AnnotationControls({ 
  isSegmenting, 
  onSegmentStart, 
  currentSegmentDuration,
  onSegmentEnd,
  onCancelSegment,
  onCreateAnnotation
}: AnnotationControlsProps) {
  const [label, setLabel] = useState('');
  const [notes, setNotes] = useState('');
  const [showAnnotationForm, setShowAnnotationForm] = useState(false);

  const handleSegmentEnd = () => {
    if (label.trim()) {
      onCreateAnnotation(0, currentSegmentDuration, label.trim(), notes.trim() || undefined);
      setLabel('');
      setNotes('');
      setShowAnnotationForm(false);
    }
    onSegmentEnd('SEGMENT');
  };

  if (isSegmenting) {
    return (
      <div className="px-4 py-2 space-y-4">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2 text-white/90">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
            <span className="text-sm">Recording segment ({currentSegmentDuration.toFixed(2)}s)</span>
          </div>
        </div>
        
        {showAnnotationForm && (
          <div className="space-y-3 p-3 bg-gray-100 rounded-lg">
            <div>
              <Label htmlFor="annotation-label">Label *</Label>
              <Input
                id="annotation-label"
                value={label}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setLabel(e.target.value)}
                placeholder="Enter annotation label"
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="annotation-notes">Notes (optional)</Label>
              <textarea
                id="annotation-notes"
                value={notes}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setNotes(e.target.value)}
                placeholder="Enter additional notes"
                className="mt-1 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                rows={2}
              />
            </div>
          </div>
        )}
        
        <div className="flex items-center gap-2"> 
          <Button 
            size="sm" 
            onClick={handleSegmentEnd}
            className="bg-red-600 hover:bg-red-700 h-8 text-sm flex-1 text-white" 
            title="End segment"
            disabled={!label.trim()}
          >
            <Pause className="h-4 w-4 mr-1" />
            End Segment (Enter)
          </Button>
          <Button 
            size="sm" 
            variant="ghost"
            className="text-gray-300 hover:text-white hover:bg-white/10 h-8"
            onClick={() => setShowAnnotationForm(!showAnnotationForm)}
            title="Toggle annotation form"
          >
            {showAnnotationForm ? 'Hide Form' : 'Add Details'}
          </Button>
          <Button 
            size="sm" 
            variant="ghost"
            className="text-gray-300 hover:text-white hover:bg-white/10 h-8"
            onClick={onCancelSegment}
            title="Cancel segment (Esc)"
          >
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-2">
      <Button 
        onClick={onSegmentStart}
        className="w-full h-8 bg-emerald-600 hover:bg-emerald-700 text-sm text-white"
        title="Start a new segment"
      >
        <PlayCircle className="h-4 w-4 mr-1"/>
        New Segment (Enter)
      </Button>
    </div>
  );
} 