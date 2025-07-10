import React from 'react';

interface AWSUploadNotificationProps {
  isVisible: boolean;
  status: 'idle' | 'connecting' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  message: string;
  error?: string;
  awsUrl?: string;
  onClose: () => void;
}

const AWSUploadNotification: React.FC<AWSUploadNotificationProps> = ({
  isVisible,
  status,
  progress,
  message,
  error,
  awsUrl,
  onClose
}) => {
  if (!isVisible) return null;

  const getStatusIcon = () => {
    switch (status) {
      case 'connecting': return '🔗';
      case 'uploading': return '📤';
      case 'processing': return '⚙️';
      case 'completed': return '✅';
      case 'error': return '❌';
      default: return '📋';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'connecting': return 'bg-blue-500';
      case 'uploading': return 'bg-blue-600';
      case 'processing': return 'bg-yellow-500';
      case 'completed': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusTitle = () => {
    switch (status) {
      case 'connecting': return 'Connecting to AWS...';
      case 'uploading': return 'Uploading to AWS S3...';
      case 'processing': return 'Processing in AWS...';
      case 'completed': return 'Successfully uploaded to AWS!';
      case 'error': return 'Upload failed';
      default: return 'Upload Status';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 max-w-md">
      <div className={`${getStatusColor()} text-white rounded-lg shadow-lg p-4 border-l-4 border-l-white/20`}>
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3">
            <div className="text-2xl">{getStatusIcon()}</div>
            <div className="flex-1">
              <h4 className="font-semibold text-sm">{getStatusTitle()}</h4>
              <p className="text-xs text-white/90 mt-1">{message}</p>
              
              {status === 'uploading' && (
                <div className="mt-2">
                  <div className="w-full bg-white/20 rounded-full h-2">
                    <div 
                      className="bg-white h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-white/80 mt-1">
                    {Math.round(progress)}% complete
                  </p>
                </div>
              )}
              
              {status === 'completed' && awsUrl && (
                <div className="mt-2 p-2 bg-white/10 rounded">
                  <p className="text-xs text-white/90">
                    <strong>AWS S3 URL:</strong> {awsUrl}
                  </p>
                </div>
              )}
              
              {status === 'error' && error && (
                <div className="mt-2 p-2 bg-white/10 rounded">
                  <p className="text-xs text-white/90">
                    <strong>Error:</strong> {error}
                  </p>
                </div>
              )}
            </div>
          </div>
          
          <button
            onClick={onClose}
            className="text-white/70 hover:text-white text-lg font-bold"
          >
            ×
          </button>
        </div>
        
        {/* AWS Connection Info */}
        <div className="mt-3 pt-3 border-t border-white/20">
          <div className="text-xs text-white/80 space-y-1">
            <p>• AWS Region: us-east-1</p>
            <p>• S3 Bucket: spokhand-data</p>
            <p>• API Gateway: Connected</p>
            {status === 'completed' && (
              <p className="text-green-200 font-medium">• ✅ Data successfully submitted to AWS</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AWSUploadNotification; 