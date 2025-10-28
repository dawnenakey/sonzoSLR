import React from 'react';
import { Camera, Eye, Edit3, Download, CheckCircle2 } from 'lucide-react';

/**
 * Process Bar Component - Shows 4-step workflow for sign language annotation
 * @param {string} currentStep - 'record' | 'preview' | 'annotate' | 'export'
 * @param {function} onStepClick - Callback when a step is clicked
 */
export default function ProcessBar({ currentStep = 'record', onStepClick }) {
  const steps = [
    { id: 'record', label: 'Record', icon: Camera },
    { id: 'preview', label: 'Preview', icon: Eye },
    { id: 'annotate', label: 'Annotate', icon: Edit3 },
    { id: 'export', label: 'Export', icon: Download }
  ];

  const getStepStatus = (stepId) => {
    const currentIndex = steps.findIndex(s => s.id === currentStep);
    const stepIndex = steps.findIndex(s => s.id === stepId);
    
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'active';
    return 'inactive';
  };

  const getStepContent = () => {
    const content = {
      record: {
        title: 'Record Your Sign Language Video',
        instructions: [
          'Make sure your face and hands are clearly visible.',
          'Use a neutral background and good lighting.',
          'Avoid moving the camera — keep it stable and in HD.'
        ]
      },
      preview: {
        title: 'Preview Your Video',
        instructions: [
          'Review your recorded video for quality.',
          'Check lighting, clarity, and framing.',
          'Make sure signs are clearly visible.'
        ]
      },
      annotate: {
        title: 'Annotate Your Signs',
        instructions: [
          'Mark segment boundaries on the timeline.',
          'Add labels and descriptions for each sign.',
          'Include facial expressions and hand dominance details.'
        ]
      },
      export: {
        title: 'Export Your Annotations',
        instructions: [
          'Choose your export format (JSON, CSV, etc.).',
          'Download or share your annotated data.',
          'Save to cloud storage if needed.'
        ]
      }
    };
    return content[currentStep] || content.record;
  };

  const stepContent = getStepContent();
  const currentIndex = steps.findIndex(s => s.id === currentStep);

  return (
    <div className="w-full mb-8">
      {/* Process Bar */}
      <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const status = getStepStatus(step.id);
            const isActive = status === 'active';
            const isCompleted = status === 'completed';
            
            return (
              <React.Fragment key={step.id}>
                <div 
                  className="flex flex-col items-center cursor-pointer"
                  onClick={() => onStepClick && onStepClick(step.id)}
                >
                  <div 
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                      isActive 
                        ? 'bg-blue-600 text-white shadow-lg transform scale-110' 
                        : isCompleted 
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-200 text-gray-400'
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle2 className="h-6 w-6" />
                    ) : (
                      <Icon className="h-6 w-6" />
                    )}
                  </div>
                  <span 
                    className={`mt-2 text-sm font-medium transition-colors ${
                      isActive 
                        ? 'text-blue-600' 
                        : isCompleted 
                        ? 'text-green-600'
                        : 'text-gray-400'
                    }`}
                  >
                    {step.label}
                  </span>
                </div>
                
                {/* Connector Line */}
                {index < steps.length - 1 && (
                  <div 
                    className={`flex-1 h-0.5 mx-4 transition-colors ${
                      isCompleted ? 'bg-green-500' : 'bg-gray-200'
                    }`}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Current Step Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          {stepContent.title}
        </h2>
        <ul className="space-y-2">
          {stepContent.instructions.map((instruction, index) => (
            <li key={index} className="flex items-start text-gray-700">
              <span className="text-blue-600 mr-2">•</span>
              <span>{instruction}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

