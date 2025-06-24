# Local Annotation System

This document describes the new local annotation system that replaces the Base44 integration, providing a self-contained, fully functional annotation platform.

## Features

### ✅ User Management
- Local user authentication (demo mode)
- User sessions and preferences
- Role-based access (annotator, admin, viewer)

### ✅ Session Management
- Create and manage annotation sessions
- Organize videos by session
- Session status tracking (active, completed, archived)

### ✅ Video Management
- Upload videos to sessions
- Video metadata tracking
- Local video storage using blob URLs

### ✅ Annotation System
- Create, edit, and delete annotations
- Time-based annotation with start/end times
- Label and notes support
- Confidence scoring
- Hand shape and location tracking

### ✅ AI Integration
- Mock AI analysis for demonstration
- Automatic annotation generation
- Confidence scoring for AI-generated annotations

### ✅ Data Management
- Export annotations to JSON
- Import annotations from JSON
- Local storage persistence
- Data backup and restore

## Architecture

### Local Storage
The system uses browser localStorage for data persistence:
- `spokhand_sessions`: Session data
- `spokhand_users`: User data  
- `spokhand_annotations`: Annotation data
- `video_*`: Video blob URLs

### Components
- `LocalAnnotationClient`: Main client class
- `App.tsx`: Updated to use local system
- `AnnotationControls`: Enhanced with annotation creation
- `AnnotationList`: Updated with new interface

## Usage

### Getting Started
1. The app automatically logs in with demo credentials
2. Create a new session or select existing one
3. Upload video files
4. Start annotating with the timeline controls

### Creating Annotations
1. Click "New Segment" to start recording
2. Click "Add Details" to enter label and notes
3. Click "End Segment" to save the annotation

### AI Analysis
1. Upload a video
2. Click "AI Analyze" to run mock analysis
3. Review and edit generated annotations

### Data Export/Import
1. Click "Export Data" to download JSON file
2. Click "Import Data" to restore from JSON file

## Benefits Over Base44

### ✅ No External Dependencies
- Works offline
- No API keys required
- No network dependencies

### ✅ Faster Development
- Immediate feedback
- No API integration delays
- Full control over features

### ✅ Better UX
- Instant response times
- No loading states for API calls
- Consistent interface

### ✅ Data Privacy
- All data stays local
- No external data sharing
- Full data ownership

## Future Enhancements

### Real AI Integration
- Connect to your ML models
- Replace mock analysis with real predictions
- Add model confidence scoring

### Cloud Sync
- Optional cloud storage
- Multi-device synchronization
- Team collaboration features

### Advanced Features
- Video preprocessing
- Batch annotation
- Advanced search and filtering
- Annotation validation workflows

## Migration from Base44

The system is designed to be a drop-in replacement for Base44:
- Same annotation interface
- Compatible data structures
- Export/import capabilities for data migration

## Technical Details

### Data Structure
```typescript
interface Annotation {
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
```

### API Methods
- `signIn(email, password)`: User authentication
- `createSession(name, description)`: Create new session
- `uploadVideo(sessionId, file)`: Upload video
- `createAnnotation(videoId, annotation)`: Create annotation
- `analyzeVideo(videoId)`: Run AI analysis
- `exportData()`: Export all data
- `importData(jsonData)`: Import data

## Troubleshooting

### Data Loss
- Data is stored in localStorage
- Clear browser data will remove annotations
- Use export feature for backups

### Performance
- Large video files may impact performance
- Consider video compression for better UX
- Monitor localStorage size limits

### Browser Compatibility
- Requires modern browser with localStorage support
- Tested on Chrome, Firefox, Safari, Edge
- Mobile browsers may have limitations 