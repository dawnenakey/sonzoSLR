import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { Download } from 'lucide-react';

interface Annotation {
  id: string;
  startTime: number;
  endTime: number;
  label: string;
  notes?: string;
}

interface ExportJsonDialogProps {
  isOpen: boolean;
  onClose: () => void;
  annotations: Annotation[];
  videoMetadata?: {
    name: string;
    duration: number;
  };
}

export default function ExportJsonDialog({
  isOpen,
  onClose,
  annotations,
  videoMetadata
}: ExportJsonDialogProps) {
  const handleExport = () => {
    const data = {
      version: "1.0",
      video: videoMetadata,
      annotations: annotations.map(({ id, ...rest }) => ({
        ...rest,
        duration: rest.endTime - rest.startTime
      }))
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `annotations_${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Export Annotations</DialogTitle>
          <DialogDescription>
            Export your annotations as a JSON file
          </DialogDescription>
        </DialogHeader>
        <div className="mt-4 space-y-4">
          <div className="rounded-lg bg-muted p-4">
            <p className="text-sm font-medium">Summary</p>
            <ul className="mt-2 text-sm text-muted-foreground">
              <li>Total annotations: {annotations.length}</li>
              {videoMetadata && (
                <li>Video duration: {videoMetadata.duration.toFixed(2)}s</li>
              )}
            </ul>
          </div>
          <Button
            className="w-full"
            onClick={handleExport}
          >
            <Download className="mr-2 h-4 w-4" />
            Export JSON
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
} 