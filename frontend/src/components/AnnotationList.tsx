import React from 'react';
import { Button } from "./ui/button";
import { Edit2, Trash2 } from 'lucide-react';

interface Annotation {
  id: string;
  startTime: number;
  endTime: number;
  label: string;
  notes?: string;
}

interface AnnotationListProps {
  annotations: Annotation[];
  onEdit: (annotation: Annotation) => void;
  onDelete: (id: string) => void;
  onSelect: (annotation: Annotation) => void;
  selectedAnnotationId?: string;
}

export default function AnnotationList({
  annotations,
  onEdit,
  onDelete,
  onSelect,
  selectedAnnotationId
}: AnnotationListProps) {
  return (
    <div className="space-y-2">
      {annotations.map((annotation) => (
        <div
          key={annotation.id}
          className={`p-3 rounded-lg border ${
            selectedAnnotationId === annotation.id
              ? 'bg-accent border-accent'
              : 'bg-card border-border hover:border-accent/50'
          } cursor-pointer transition-colors`}
          onClick={() => onSelect(annotation)}
        >
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-sm">{annotation.label}</h4>
              <p className="text-xs text-muted-foreground">
                {annotation.startTime.toFixed(2)}s - {annotation.endTime.toFixed(2)}s
              </p>
              {annotation.notes && (
                <p className="text-xs text-muted-foreground mt-1">
                  {annotation.notes}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0"
                onClick={(e) => {
                  e.stopPropagation();
                  onEdit(annotation);
                }}
              >
                <Edit2 className="h-4 w-4" />
                <span className="sr-only">Edit</span>
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 hover:bg-destructive/10 hover:text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(annotation.id);
                }}
              >
                <Trash2 className="h-4 w-4" />
                <span className="sr-only">Delete</span>
              </Button>
            </div>
          </div>
        </div>
      ))}
      {annotations.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          No annotations yet
        </div>
      )}
    </div>
  );
} 