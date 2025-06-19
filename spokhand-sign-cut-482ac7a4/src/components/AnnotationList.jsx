import React from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { formatTime } from './timeUtils';
import { HandMetal, Pause, CornerUpRight, Pencil, Trash2, Play, MoreVertical, Download as DownloadIcon, Trash as TrashAllIcon, AlertCircle } from 'lucide-react';
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function AnnotationList({ 
  annotations, 
  onEditAnnotation, 
  onDeleteAnnotation, 
  onAnnotationClick, 
  activeSegment,
  onTriggerExportJson,
  onTriggerDeleteAll
}) {
  const sortedAnnotations = [...annotations].sort((a, b) => a.start_time - b.start_time);

  const getSegmentIcon = (type) => {
    switch (type) {
      case 'SIGN_UNIT':
        return <HandMetal className="h-4 w-4" />;
      case 'BREAK':
        return <Pause className="h-4 w-4" />;
      case 'TRANSITION':
        return <CornerUpRight className="h-4 w-4" />;
      case 'FALSE_POSITIVE':
        return <AlertCircle className="h-4 w-4" />;
      default:
        return <Play className="h-4 w-4" />; // Default for 'SEGMENT' and others
    }
  };

  const getSegmentBadge = (type) => {
    let className = '';
    let label = '';
    
    switch (type) {
      case 'SIGN_UNIT':
        className = 'bg-blue-100 text-blue-800 border-blue-200';
        label = 'Sign Unit';
        break;
      case 'BREAK':
        className = 'bg-amber-100 text-amber-800 border-amber-200';
        label = 'Break';
        break;
      case 'TRANSITION':
        className = 'bg-purple-100 text-purple-800 border-purple-200';
        label = 'Transition';
        break;
      case 'FALSE_POSITIVE':
        className = 'bg-gray-100 text-gray-800 border-gray-200';
        label = 'False Positive';
        break;
      default:
        className = 'bg-blue-100 text-blue-800 border-blue-200';
        label = 'Segment';
    }
    
    return (
      <Badge variant="outline" className={className}>
        <span className="flex items-center gap-1">
          {getSegmentIcon(type)}
          {label}
        </span>
      </Badge>
    );
  };
  
  const handleRowClick = (annotation) => {
    if (onAnnotationClick) {
      onAnnotationClick(annotation);
    }
  };

  return (
    <div className="bg-white p-4 sm:p-6 rounded-xl shadow-lg">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-medium text-gray-900">Annotations ({sortedAnnotations.length})</h3>
        {annotations.length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <MoreVertical className="h-4 w-4" />
                <span className="sr-only">Annotation Options</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onTriggerExportJson} className="flex items-center gap-2 cursor-pointer">
                <DownloadIcon className="h-4 w-4" />
                Export as JSON
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                onClick={onTriggerDeleteAll} 
                className="flex items-center gap-2 text-red-600 hover:!text-red-700 hover:!bg-red-50 cursor-pointer"
              >
                <TrashAllIcon className="h-4 w-4" />
                Delete All Annotations
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
      
      {annotations.length === 0 ? (
        <div className="h-24 flex items-center justify-center bg-gray-50 rounded-lg border border-dashed border-gray-300">
          <p className="text-gray-500 text-sm">No annotations yet. Start segmenting the video.</p>
        </div>
      ) : (
        <div className="overflow-y-auto max-h-[400px]">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">#</TableHead>
                <TableHead className="w-[65%]">Type & Segment Info</TableHead>
                <TableHead className="text-right w-[100px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedAnnotations.map((annotation, index) => {
                const isActive = activeSegment && activeSegment.id === annotation.id;
                const isQualified = annotation.qualified;
                
                return (
                  <TableRow 
                    key={annotation.id || index}
                    className={`${isActive ? 'bg-emerald-50' : ''} ${isQualified ? 'bg-blue-50/50' : ''} hover:bg-gray-50/80 cursor-pointer`}
                    onClick={() => handleRowClick(annotation)}
                  >
                    <TableCell className="px-2 py-2">
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className={`h-7 w-7 ${isActive ? 'bg-emerald-100 text-emerald-800' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation(); 
                          handleRowClick(annotation);
                        }}
                      >
                        <span className="sr-only">Edit segment</span>
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-gray-100 text-xs">
                          {index + 1}
                        </span>
                      </Button>
                    </TableCell>
                    <TableCell className="py-2">
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          {getSegmentBadge(annotation.segment_type)}
                          <span className="text-xs text-gray-500 font-mono">
                            {formatTime(annotation.start_time)} → {formatTime(annotation.start_time + annotation.duration)} ({annotation.duration.toFixed(1)}s)
                          </span>
                        </div>
                        {(annotation.description || annotation.label) && (
                          <span className="text-xs text-gray-600 italic truncate max-w-[20rem] sm:max-w-xs md:max-w-sm lg:max-w-md" title={annotation.description || annotation.label}>
                            "{annotation.label || annotation.description}"
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right px-2 py-1">
                      <div className="flex justify-end gap-1">
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-7 w-7 hover:bg-gray-100"
                          onClick={(e) => {
                            e.stopPropagation(); 
                            onEditAnnotation(annotation);
                          }}
                        >
                          <Pencil className="h-3.5 w-3.5" />
                          <span className="sr-only">Edit</span>
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-7 w-7 hover:bg-gray-100 hover:text-red-600"
                          onClick={(e) => {
                            e.stopPropagation(); 
                            onDeleteAnnotation(annotation);
                          }}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                          <span className="sr-only">Delete</span>
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}