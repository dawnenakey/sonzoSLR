import React from 'react';
import { Button } from "@/components/ui/button";
import { PlayCircle, Pause } from 'lucide-react';

export default function AnnotationControls({ 
  isSegmenting, 
  onSegmentStart, 
  currentSegmentDuration,
  onSegmentEnd,
  onCancelSegment
}) {
  if (isSegmenting) {
    return (
      <div className="px-4 py-2">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2 text-white/90">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
            <span className="text-sm">Recording segment ({currentSegmentDuration.toFixed(2)}s)</span>
          </div>
        </div>
        
        <div className="flex items-center gap-2"> 
          <Button 
            size="sm" 
            onClick={() => onSegmentEnd('SEGMENT')}
            className="bg-red-600 hover:bg-red-700 h-8 text-sm flex-1 text-white" 
            title="End segment"
          >
            <Pause className="h-4 w-4 mr-1" />
            End Segment (Enter)
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