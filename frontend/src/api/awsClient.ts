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

    return response.json();
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
    const formData = new FormData();
    formData.append('video', file);

    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/upload-video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload video: ${response.statusText}`);
    }

    return response.json();
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

    return response.json();
  },
};

// Main API client
export const awsAPI = {
  sessions: sessionAPI,
  videos: videoAPI,
  annotations: annotationAPI,
};

export default awsAPI; 