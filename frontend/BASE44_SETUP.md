# Base44 API Integration Setup

## Current Status
The application is currently using a **mock implementation** of the Base44 API. To integrate with the real Base44 API, follow these steps:

## Step 1: Get Base44 Credentials

Contact the Base44 team to obtain:
1. **API Key** - For authenticating API requests
2. **Base URL** - The endpoint for the Base44 API
3. **SDK Package** - Access to the `@base44/sdk` package
4. **Authentication Credentials** - User account or service account details

## Step 2: Set Environment Variables

Create a `.env` file in the `frontend` directory with your credentials:

```bash
# Base44 API Configuration
VITE_BASE44_API_KEY=your_actual_api_key_here
VITE_BASE44_AUTH_TOKEN=your_actual_auth_token_here
VITE_BASE44_BASE_URL=https://api.base44.com
VITE_BASE44_APP_ID=6827319154cb6b61482ac7a4
```

## Step 3: Install Base44 SDK

Once you have access to the Base44 SDK:

```bash
npm install @base44/sdk
```

## Step 4: Replace Mock Implementation

Update `src/api/base44Client.ts`:

1. Uncomment the real SDK import:
```typescript
import { createClient } from '@base44/sdk';
```

2. Replace the mock client with the real one:
```typescript
export const base44 = createClient({
  appId: BASE44_CONFIG.appId,
  apiKey: BASE44_CONFIG.apiKey,
  baseUrl: BASE44_CONFIG.baseUrl,
  requiresAuth: true
});
```

3. Remove the mock implementation class

## Step 5: Test the Integration

1. Start the development server:
```bash
npm start
```

2. Check the browser console for Base44 API calls
3. Test authentication flow
4. Test video upload and annotation creation

## Current Features Available

The mock implementation supports:
- ✅ User authentication (mock)
- ✅ Video creation and management
- ✅ Annotation creation, update, delete
- ✅ File upload (mock)
- ✅ LLM integration (mock)
- ✅ Email sending (mock)
- ✅ Image generation (mock)

## API Endpoints Available

Once integrated, you'll have access to:

### Authentication
- `base44.auth.signIn(email, password)`
- `base44.auth.signOut()`
- `base44.auth.getStatus()`

### Video Management
- `base44.entities.Video.create(data)`
- `base44.entities.Video.update(id, data)`
- `base44.entities.Video.delete(id)`
- `base44.entities.Video.list(filters)`
- `base44.entities.Video.get(id)`

### Annotation Management
- `base44.entities.Annotation.create(data)`
- `base44.entities.Annotation.update(id, data)`
- `base44.entities.Annotation.delete(id)`
- `base44.entities.Annotation.list(filters)`

### Integrations
- `base44.integrations.Core.UploadFile(file, options)`
- `base44.integrations.Core.InvokeLLM(prompt, options)`
- `base44.integrations.Core.SendEmail(to, subject, body)`
- `base44.integrations.Core.GenerateImage(prompt, options)`
- `base44.integrations.Core.ExtractDataFromUploadedFile(fileId, options)`

## Troubleshooting

1. **SDK not found**: Make sure you have access to the `@base44/sdk` package
2. **Authentication failed**: Check your API key and auth token
3. **API errors**: Verify the base URL and app ID
4. **CORS issues**: Ensure the Base44 API allows requests from your domain

## Support

For Base44 API support, contact the Base44 team directly.
For application integration issues, check the browser console for error messages. 