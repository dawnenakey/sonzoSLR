import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

interface AnnotationDetailDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: { label: string; notes: string }) => void;
  initialData?: { label: string; notes: string };
}

export default function AnnotationDetailDialog({
  isOpen,
  onClose,
  onSave,
  initialData = { label: '', notes: '' }
}: AnnotationDetailDialogProps) {
  const [label, setLabel] = useState(initialData.label);
  const [notes, setNotes] = useState(initialData.notes);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave({ label, notes });
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Annotation Details</DialogTitle>
            <DialogDescription>
              Add details about this sign segment.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="label" className="text-right">
                Sign Label
              </Label>
              <Input
                id="label"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                className="col-span-3"
                required
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="notes" className="text-right">
                Notes
              </Label>
              <Input
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="submit">Save</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
} 