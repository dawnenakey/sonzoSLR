import React, { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

interface ImportVideoDialogProps {
  onFileSelect: (file: File) => void;
  disabled: boolean;
}

export function ImportVideoDialog({ onFileSelect, disabled }: ImportVideoDialogProps) {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleImport = () => {
    if (file) {
      onFileSelect(file);
      // Reset file input after loading
      setFile(null);
      const input = document.getElementById('video-file') as HTMLInputElement;
      if (input) {
        input.value = '';
      }
    }
  };

  return (
    <div className="flex items-center justify-center gap-2">
      <Label htmlFor="video-file" className="sr-only">Video File</Label>
      <Input
        id="video-file"
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="max-w-xs"
        disabled={disabled}
      />
      <Button onClick={handleImport} disabled={!file || disabled}>
        Load Video
      </Button>
    </div>
  );
}