import React, { useCallback } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { Upload } from 'lucide-react';

interface ImportVideoDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (file: File) => void;
}

export default function ImportVideoDialog({
  isOpen,
  onClose,
  onImport
}: ImportVideoDialogProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('video/')) {
        onImport(file);
        onClose();
      }
    },
    [onImport, onClose]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onImport(file);
        onClose();
      }
    },
    [onImport, onClose]
  );

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Import Video</DialogTitle>
          <DialogDescription>
            Drag and drop a video file or click to browse
          </DialogDescription>
        </DialogHeader>
        <div
          className="mt-4 border-2 border-dashed rounded-lg p-8 text-center hover:bg-accent/5 transition-colors cursor-pointer"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
          <p className="mt-2 text-sm text-muted-foreground">
            Drag and drop your video here, or click to select a file
          </p>
          <input
            id="file-input"
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleFileSelect}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
} 