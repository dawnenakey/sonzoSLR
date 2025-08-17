# Epic 1 Setup Guide - SPOKHAND SIGNCUT Authentication System

## Overview
This guide will help you set up and test Epic 1 (Global Platform Setup) for SPOKHAND SIGNCUT, which includes:
- JWT authentication, CRUD audit logs, API rate limiting
- RBAC with cumulative roles: Translator, Segmenter, Qualifier, Expert, Admin
- DB schema for datasets, providers, videos, text corpora, segments, lexicon, users, AI artifacts, gamification, exports

## Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Auth API      │    │   DynamoDB      │
│   (React)       │◄──►│   (Flask)       │◄──►│   (Users +      │
│                 │    │                 │    │   Audit Logs)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites
- Python 3.9+
- AWS CLI configured with appropriate permissions
- Node.js 16+ (for frontend)
- Access to AWS DynamoDB

## Installation Steps

### 1. Install Python Dependencies
```bash
cd src/
pip install -r ../requirements-auth.txt
```

### 2. Set Environment Variables
Create a `.env` file in the `src/` directory:
```bash
# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# AWS Configuration
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### 3. Setup DynamoDB Tables
```bash
cd src/
python setup_database.py
```

This will create:
- `spokhand-users` table for user management
- `spokhand-audit-logs` table for audit trails
- Sample users with different roles

## Running the System

### 1. Start Authentication Service
```bash
cd src/
python auth_api.py
```

The service will start on `http://localhost:5001`

### 2. Test the Backend
```bash
python test_auth.py
```

This will run comprehensive tests to verify:
- User registration
- User login/logout
- JWT token validation
- Protected endpoints
- Rate limiting
- Role-based access

## Testing the System

### Manual API Testing

#### 1. User Registration
```bash
curl -X POST http://localhost:5001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123",
    "full_name": "Test User"
  }'
```

#### 2. User Login
```bash
curl -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123"
  }'
```

#### 3. Access Protected Endpoint
```bash
curl -X GET http://localhost:5001/api/auth/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Sample Users Created
The setup script creates these test users:

| Email | Password | Roles |
|-------|----------|-------|
| `admin@spokhand.com` | `admin123456` | Admin |
| `expert@spokhand.com` | `expert123456` | Expert, Qualifier, Segmenter, Translator |
| `qualifier@spokhand.com` | `qualifier123456` | Qualifier, Segmenter, Translator |
| `segmenter@spokhand.com` | `segmenter123456` | Segmenter, Translator |
| `translator@spokhand.com` | `translator123456` | Translator |

## Role Hierarchy
```
Admin (5) > Expert (4) > Qualifier (3) > Segmenter (2) > Translator (1)
```

Each role includes permissions from all lower roles (cumulative permissions).

## Frontend Integration

### 1. Update App.tsx
Wrap your app with the AuthProvider:
```tsx
import { AuthProvider } from './contexts/AuthContext';

function App() {
  return (
    <AuthProvider>
      {/* Your existing app content */}
    </AuthProvider>
  );
}
```

### 2. Add Authentication Routes
```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Auth from './pages/Auth';
import { ProtectedRoute } from './components/auth/ProtectedRoute';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/auth" element={<Auth />} />
        <Route path="/" element={
          <ProtectedRoute>
            <Home />
          </ProtectedRoute>
        } />
        <Route path="/admin" element={
          <ProtectedRoute requiredRole="admin">
            <AdminPanel />
          </ProtectedRoute>
        } />
      </Routes>
    </BrowserRouter>
  );
}
```

### 3. Use Authentication in Components
```tsx
import { useAuth } from '../contexts/AuthContext';

function MyComponent() {
  const { user, isAuthenticated, logout } = useAuth();
  
  if (!isAuthenticated) {
    return <div>Please log in</div>;
  }
  
  return (
    <div>
      <h1>Welcome, {user.full_name}!</h1>
      <p>Your roles: {user.roles.join(', ')}</p>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

## Development Priority Roadmap

### Phase 1: Core Infrastructure (2-3 weeks)
1. **User Authentication System**
   - JWT implementation
   - Role-based access control
   - User management interface

2. **Text Corpus Management**
   - Text editor for translators
   - Approval workflow
   - Link to video creation

### Phase 2: AI Integration (3-4 weeks)
1. **AI Block System**
   - AI-generated segment suggestions
   - Validation/invalidation workflow
   - Confidence scoring display

2. **Enhanced Qualification**
   - Linguistic metadata forms
   - AI gloss suggestions
   - Pose/depth preview integration

### Phase 3: Advanced Features (4-5 weeks)
1. **Role-Based Dashboards**
   - Translator, Segmenter, Qualifier views
   - KPI tracking and metrics
   - Task assignment system

2. **Export & API System**
   - Full JSON schema implementation
   - Pose/depth data export
   - External API endpoints

### Phase 4: Gamification & Admin (2-3 weeks)
1. **Points & Leaderboards**
2. **Admin Panel**
3. **Advanced Analytics**

## Immediate Next Steps

1. **Create Authentication Backend**
   - JWT token generation/validation
   - User login/registration endpoints
   - Password hashing and security

2. **Build User Management Frontend**
   - Login/register forms
   - User profile management
   - Role assignment interface

3. **Implement Role-Based Routing**
   - Protected route components
   - Role-specific navigation
   - Permission-based UI rendering

**Your current platform is excellent for video annotation, but Epic 1 needs to be built from scratch.** The good news is that your existing AWS infrastructure and frontend architecture will make this implementation much easier than starting from zero.

## Monitoring & Debugging

### 1. Check DynamoDB Tables
```bash
aws dynamodb scan --table-name spokhand-users
aws dynamodb scan --table-name spokhand-audit-logs
```

### 2. View Audit Logs
The system automatically logs:
- User registration
- User login/logout
- Role changes
- Token refresh
- Failed authentication attempts

### 3. Check API Logs
Monitor the Flask application console for:
- Request/response logs
- Error messages
- Rate limiting events

## Security Considerations

### 1. JWT Secret
- **NEVER** commit the JWT secret to version control
- Use environment variables or AWS Secrets Manager
- Rotate secrets regularly in production

### 2. Password Security
- Passwords are hashed using bcrypt
- Minimum 8 characters required
- Consider adding password complexity requirements

### 3. Rate Limiting
- Currently set to 100 requests per hour per user
- Adjust based on your application needs
- Consider using Redis for production rate limiting

### 4. CORS Configuration
- Add proper CORS headers for production
- Restrict origins to your frontend domains

## Production Deployment

### 1. AWS Lambda
Package the authentication service for AWS Lambda:
```bash
pip install -r requirements-auth.txt -t package/
cp auth_service.py package/
cp auth_api.py package/
cd package
zip -r ../auth-lambda.zip .
```

### 2. Environment Variables
Set these in your Lambda function:
- `JWT_SECRET`
- `DYNAMODB_TABLE_PREFIX`
- `ENVIRONMENT`

### 3. IAM Permissions
Ensure your Lambda has permissions for:
- DynamoDB read/write access
- CloudWatch Logs
- Any other AWS services you use

## Next Steps

After Epic 1 is working:

1. **Epic 2**: Text Corpus Management
2. **Epic 3**: Enhanced Video Workspace
3. **Epic 4**: AI Integration
4. **Epic 5**: Lexicon Management

## Support

If you encounter issues:
1. Check the test output for specific error messages
2. Verify all dependencies are installed
3. Check AWS credentials and permissions
4. Review the audit logs for authentication failures

---

**Congratulations!** You've successfully implemented Epic 1 of SPOKHAND SIGNCUT. The platform now has a solid foundation for user management and role-based access control. 