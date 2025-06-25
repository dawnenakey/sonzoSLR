import { API_CONFIG } from '../config/api';

const API_BASE_URL = API_CONFIG.BASE_URL;

export interface Session {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  status: 'active' | 'completed' | 'archived';
}

export interface Video {
  id: string;
  sessionId: string;
  filename: string;
  size: number;
  duration?: number;
  uploadedAt: string;
  status: 'uploading' | 'processing' | 'ready' | 'error';
}

export interface Annotation {
  id: string;
  videoId: string;
  startTime: number;
  endTime: number;
  label: string;
  confidence?: number;
  notes?: string;
  createdAt: string;
  updatedAt: string;
}

// Session API calls
export const sessionAPI = {
  // Create a new session
  create: async (data: { name: string; description?: string }): Promise<Session> => {
    const response = await fetch(`${API_BASE_URL}/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Handle the backend response format: {success: true, session_id: "..."}
    if (result.success && result.session_id) {
      return {
        id: result.session_id,
        name: data.name,
        description: data.description,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        status: 'active' as const,
      };
    } else {
      // Fallback to assuming it's a direct Session object
      return result;
    }
  },

  // Get a session by ID
  get: async (sessionId: string): Promise<Session> => {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`);

    if (!response.ok) {
      throw new Error(`Failed to get session: ${response.statusText}`);
    }

    return response.json();
  },

  // Upload video to a session
  uploadVideo: async (sessionId: string, file: File): Promise<Video> => {
    // First, get a presigned URL for upload
    const presignedResponse = await fetch(`${API_BASE_URL}/sessions/${sessionId}/upload-video`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        filename: file.name,
        contentType: file.type || 'video/mp4'
      }),
    });

    if (!presignedResponse.ok) {
      throw new Error(`Failed to get upload URL: ${presignedResponse.statusText}`);
    }

    const presignedData = await presignedResponse.json();
    if (!presignedData.success) {
      throw new Error(`Failed to get upload URL: ${presignedData.error}`);
    }

    // Upload the file directly to S3 using the presigned URL
    const uploadResponse = await fetch(presignedData.uploadUrl, {
      method: 'PUT',
      headers: {
        'Content-Type': file.type || 'video/mp4',
      },
      body: file,
    });

    if (!uploadResponse.ok) {
      throw new Error(`Failed to upload video to S3: ${uploadResponse.statusText}`);
    }

    // Return the video object from the backend response
    if (presignedData.video) {
      return presignedData.video;
    } else {
      // Fallback to constructing the video object
      return {
        id: presignedData.objectKey,
        sessionId,
        filename: file.name,
        size: file.size,
        uploadedAt: new Date().toISOString(),
        status: 'ready' as const,
      };
    }
  },
};

// Video API calls
export const videoAPI = {
  // Get video stream URL
  getStreamUrl: (videoId: string): string => {
    return `${API_BASE_URL}/videos/${videoId}/stream`;
  },

  // Get video metadata
  get: async (videoId: string): Promise<Video> => {
    const response = await fetch(`${API_BASE_URL}/videos/${videoId}`);

    if (!response.ok) {
      throw new Error(`Failed to get video: ${response.statusText}`);
    }

    return response.json();
  },
};

// Annotation API calls
export const annotationAPI = {
  // Create a new annotation
  create: async (data: {
    videoId: string;
    startTime: number;
    endTime: number;
    label: string;
    confidence?: number;
    notes?: string;
  }): Promise<Annotation> => {
    const response = await fetch(`${API_BASE_URL}/videos/${data.videoId}/annotations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Failed to create annotation: ${response.statusText}`);
    }

    return response.json();
  },

  // Update an annotation
  update: async (annotationId: string, data: Partial<Annotation>): Promise<Annotation> => {
    const response = await fetch(`${API_BASE_URL}/annotations/${annotationId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Failed to update annotation: ${response.statusText}`);
    }

    return response.json();
  },

  // Delete an annotation
  delete: async (annotationId: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/annotations/${annotationId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to delete annotation: ${response.statusText}`);
    }
  },

  // Get annotations for a video
  getByVideo: async (videoId: string): Promise<Annotation[]> => {
    const response = await fetch(`${API_BASE_URL}/videos/${videoId}/annotations`);

    if (!response.ok) {
      throw new Error(`Failed to get annotations: ${response.statusText}`);
    }

    const data = await response.json();
    
    // Handle the success response format from the backend
    if (data.success && data.annotations) {
      return data.annotations;
    } else {
      // Fallback to assuming the response is directly an array
      return data;
    }
  },
};

// Main API client
export const awsAPI = {
  sessions: sessionAPI,
  videos: videoAPI,
  annotations: annotationAPI,
};

export default awsAPI; 