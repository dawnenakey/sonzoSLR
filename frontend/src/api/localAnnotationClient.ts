// Local Annotation Client - Self-contained annotation system
// Replaces Base44 integration with local storage and modern web APIs

export interface Annotation {
  id: string;
  videoId: string;
  startTime: number;
  endTime: number;
  label: string;
  confidence: number;
  handShape?: string;
  location?: string;
  notes?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Video {
  id: string;
  filename: string;
  duration: number;
  resolution: string;
  uploadDate: string;
  annotations: Annotation[];
  thumbnail?: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'annotator' | 'admin' | 'viewer';
}

export interface AnnotationSession {
  id: string;
  name: string;
  description?: string;
  videos: Video[];
  createdAt: string;
  updatedAt: string;
  status: 'active' | 'completed' | 'archived';
}

class LocalAnnotationClient {
  private storageKey = 'spokhand_annotations';
  private usersKey = 'spokhand_users';
  private sessionsKey = 'spokhand_sessions';

  constructor() {
    this.initializeStorage();
  }

  private initializeStorage() {
    if (!localStorage.getItem(this.storageKey)) {
      localStorage.setItem(this.storageKey, JSON.stringify([]));
    }
    if (!localStorage.getItem(this.usersKey)) {
      localStorage.setItem(this.usersKey, JSON.stringify([]));
    }
    if (!localStorage.getItem(this.sessionsKey)) {
      localStorage.setItem(this.sessionsKey, JSON.stringify([]));
    }
  }

  // User Management
  async signIn(email: string, password: string): Promise<{ success: boolean; user?: User; error?: string }> {
    try {
      const users = JSON.parse(localStorage.getItem(this.usersKey) || '[]');
      const user = users.find((u: User) => u.email === email);
      
      if (user) {
        // In a real app, you'd hash and verify the password
        localStorage.setItem('current_user', JSON.stringify(user));
        return { success: true, user };
      } else {
        // Create new user for demo purposes
        const newUser: User = {
          id: this.generateId(),
          email,
          name: email.split('@')[0],
          role: 'annotator'
        };
        users.push(newUser);
        localStorage.setItem(this.usersKey, JSON.stringify(users));
        localStorage.setItem('current_user', JSON.stringify(newUser));
        return { success: true, user: newUser };
      }
    } catch (error) {
      return { success: false, error: 'Authentication failed' };
    }
  }

  async signOut(): Promise<void> {
    localStorage.removeItem('current_user');
  }

  getCurrentUser(): User | null {
    const userStr = localStorage.getItem('current_user');
    return userStr ? JSON.parse(userStr) : null;
  }

  // Session Management
  async createSession(name: string, description?: string): Promise<AnnotationSession> {
    const session: AnnotationSession = {
      id: this.generateId(),
      name,
      description,
      videos: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'active'
    };

    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    sessions.push(session);
    localStorage.setItem(this.sessionsKey, JSON.stringify(sessions));

    return session;
  }

  async getSessions(): Promise<AnnotationSession[]> {
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    return sessions;
  }

  async getSession(id: string): Promise<AnnotationSession | null> {
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    return sessions.find((s: AnnotationSession) => s.id === id) || null;
  }

  // Video Management
  async uploadVideo(sessionId: string, file: File): Promise<Video> {
    const video: Video = {
      id: this.generateId(),
      filename: file.name,
      duration: 0, // Will be calculated when video loads
      resolution: 'Unknown',
      uploadDate: new Date().toISOString(),
      annotations: []
    };

    // Store video file in IndexedDB or as blob URL
    const videoUrl = URL.createObjectURL(file);
    localStorage.setItem(`video_${video.id}`, videoUrl);

    // Update session
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    const sessionIndex = sessions.findIndex((s: AnnotationSession) => s.id === sessionId);
    if (sessionIndex !== -1) {
      sessions[sessionIndex].videos.push(video);
      sessions[sessionIndex].updatedAt = new Date().toISOString();
      localStorage.setItem(this.sessionsKey, JSON.stringify(sessions));
    }

    return video;
  }

  async getVideos(sessionId: string): Promise<Video[]> {
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    const session = sessions.find((s: AnnotationSession) => s.id === sessionId);
    return session?.videos || [];
  }

  // Annotation Management
  async createAnnotation(videoId: string, annotation: Omit<Annotation, 'id' | 'createdAt' | 'updatedAt'>): Promise<Annotation> {
    const newAnnotation: Annotation = {
      ...annotation,
      id: this.generateId(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    // Update video annotations
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    for (const session of sessions) {
      const videoIndex = session.videos.findIndex((v: Video) => v.id === videoId);
      if (videoIndex !== -1) {
        session.videos[videoIndex].annotations.push(newAnnotation);
        session.updatedAt = new Date().toISOString();
        break;
      }
    }
    localStorage.setItem(this.sessionsKey, JSON.stringify(sessions));

    return newAnnotation;
  }

  async updateAnnotation(videoId: string, annotationId: string, updates: Partial<Annotation>): Promise<Annotation> {
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    
    for (const session of sessions) {
      const videoIndex = session.videos.findIndex((v: Video) => v.id === videoId);
      if (videoIndex !== -1) {
        const annotationIndex = session.videos[videoIndex].annotations.findIndex((a: Annotation) => a.id === annotationId);
        if (annotationIndex !== -1) {
          session.videos[videoIndex].annotations[annotationIndex] = {
            ...session.videos[videoIndex].annotations[annotationIndex],
            ...updates,
            updatedAt: new Date().toISOString()
          };
          session.updatedAt = new Date().toISOString();
          localStorage.setItem(this.sessionsKey, JSON.stringify(sessions));
          return session.videos[videoIndex].annotations[annotationIndex];
        }
      }
    }

    throw new Error('Annotation not found');
  }

  async deleteAnnotation(videoId: string, annotationId: string): Promise<void> {
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    
    for (const session of sessions) {
      const videoIndex = session.videos.findIndex((v: Video) => v.id === videoId);
      if (videoIndex !== -1) {
        const annotationIndex = session.videos[videoIndex].annotations.findIndex((a: Annotation) => a.id === annotationId);
        if (annotationIndex !== -1) {
          session.videos[videoIndex].annotations.splice(annotationIndex, 1);
          session.updatedAt = new Date().toISOString();
          localStorage.setItem(this.sessionsKey, JSON.stringify(sessions));
          return;
        }
      }
    }

    throw new Error('Annotation not found');
  }

  async getAnnotations(videoId: string): Promise<Annotation[]> {
    const sessions = JSON.parse(localStorage.getItem(this.sessionsKey) || '[]');
    
    for (const session of sessions) {
      const video = session.videos.find((v: Video) => v.id === videoId);
      if (video) {
        return video.annotations;
      }
    }

    return [];
  }

  // Export/Import
  async exportData(): Promise<string> {
    const data = {
      sessions: JSON.parse(localStorage.getItem(this.sessionsKey) || '[]'),
      users: JSON.parse(localStorage.getItem(this.usersKey) || '[]'),
      exportDate: new Date().toISOString()
    };
    return JSON.stringify(data, null, 2);
  }

  async importData(jsonData: string): Promise<void> {
    const data = JSON.parse(jsonData);
    if (data.sessions) {
      localStorage.setItem(this.sessionsKey, JSON.stringify(data.sessions));
    }
    if (data.users) {
      localStorage.setItem(this.usersKey, JSON.stringify(data.users));
    }
  }

  // Utility methods
  private generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  // AI/ML Integration (Mock for now)
  async analyzeVideo(videoId: string): Promise<Annotation[]> {
    // Mock AI analysis - in a real implementation, this would call your ML model
    const mockAnnotations: Annotation[] = [
      {
        id: this.generateId(),
        videoId,
        startTime: 1.0,
        endTime: 2.5,
        label: 'HELLO',
        confidence: 0.85,
        handShape: 'A',
        location: 'neutral space',
        notes: 'Auto-detected by AI model',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      {
        id: this.generateId(),
        videoId,
        startTime: 3.0,
        endTime: 4.2,
        label: 'WORLD',
        confidence: 0.78,
        handShape: 'W',
        location: 'neutral space',
        notes: 'Auto-detected by AI model',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    ];

    // Add mock annotations to the video
    for (const annotation of mockAnnotations) {
      await this.createAnnotation(videoId, annotation);
    }

    return mockAnnotations;
  }
}

// Create and export the client instance
export const localAnnotationClient = new LocalAnnotationClient();

// Export convenience functions
export const signIn = (email: string, password: string) => localAnnotationClient.signIn(email, password);
export const signOut = () => localAnnotationClient.signOut();
export const getCurrentUser = () => localAnnotationClient.getCurrentUser();
export const createSession = (name: string, description?: string) => localAnnotationClient.createSession(name, description);
export const getSessions = () => localAnnotationClient.getSessions();
export const uploadVideo = (sessionId: string, file: File) => localAnnotationClient.uploadVideo(sessionId, file);
export const createAnnotation = (videoId: string, annotation: any) => localAnnotationClient.createAnnotation(videoId, annotation);
export const updateAnnotation = (videoId: string, annotationId: string, updates: any) => localAnnotationClient.updateAnnotation(videoId, annotationId, updates);
export const deleteAnnotation = (videoId: string, annotationId: string) => localAnnotationClient.deleteAnnotation(videoId, annotationId);
export const getAnnotations = (videoId: string) => localAnnotationClient.getAnnotations(videoId);
export const analyzeVideo = (videoId: string) => localAnnotationClient.analyzeVideo(videoId);
export const exportData = () => localAnnotationClient.exportData();
export const importData = (jsonData: string) => localAnnotationClient.importData(jsonData); 