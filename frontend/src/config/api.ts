export const API_CONFIG = {
  BASE_URL: 'https://qt8f7grhb5.execute-api.us-east-1.amazonaws.com/prod',
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
};

export const ENDPOINTS = {
  SESSIONS: '/sessions',
  VIDEOS: '/videos',
  ANNOTATIONS: '/annotations',
} as const; 