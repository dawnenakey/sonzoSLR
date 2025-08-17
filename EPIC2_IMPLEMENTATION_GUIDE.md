# Epic 2: Text Corpus Management - Implementation Guide

## 🎯 **Overview**

**Epic 2** provides comprehensive text corpus management for SPOKHAND SIGNCUT, building on the authentication foundation from Epic 1. This system enables users to create, organize, and manage text corpora that serve as the foundation for sign language data annotation and AI training.

---

## 🏗️ **Architecture**

### **System Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Epic 2 API    │    │   DynamoDB      │
│   (React)       │◄──►│   (Flask)       │◄──►│   (Text Data)   │
│                 │    │                 │    │                 │
│ • Corpus Editor │    │ • CRUD Corpus   │    │ • Text Corpora  │
│ • Segment Mgmt  │    │ • Text Segments │    │ • Text Segments │
│ • Search UI     │    │ • Search API    │    │ • Exports       │
│ • Export Tools  │    │ • Export API    │    │ • Statistics    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Data Flow**

1. **User Authentication** (Epic 1) → JWT Token
2. **Corpus Creation** → DynamoDB Storage
3. **Segment Addition** → Position-based Organization
4. **Search & Filter** → Real-time Results
5. **Export Generation** → Background Processing
6. **Statistics** → Real-time Analytics

---

## 📊 **Database Schema**

### **Text Corpora Table (`spokhand-text-corpora`)**

| Field | Type | Description | Index |
|-------|------|-------------|-------|
| `id` | String | Unique corpus identifier | Primary Key |
| `name` | String | Corpus name | - |
| `description` | String | Corpus description | - |
| `language` | String | Sign language (ASL, BSL, etc.) | GSI |
| `metadata` | JSON | Flexible metadata storage | - |
| `created_by` | String | User ID who created | GSI |
| `created_at` | String | ISO timestamp | GSI |
| `updated_at` | String | ISO timestamp | - |
| `status` | String | draft/active/archived/deleted | GSI |
| `total_segments` | Number | Count of segments | - |
| `validated_segments` | Number | Count of validated segments | - |
| `tags` | List | Array of tags | - |
| `version` | String | Semantic version | - |

### **Text Segments Table (`spokhand-text-segments`)**

| Field | Type | Description | Index |
|-------|------|-------------|-------|
| `id` | String | Unique segment identifier | Primary Key |
| `corpus_id` | String | Reference to corpus | GSI |
| `text` | String | Actual text content | - |
| `metadata` | JSON | Flexible metadata storage | - |
| `position` | Number | Order within corpus | GSI |
| `segment_type` | String | sentence/phrase/word/paragraph | GSI |
| `created_by` | String | User ID who created | - |
| `created_at` | String | ISO timestamp | - |
| `updated_at` | String | ISO timestamp | - |
| `status` | String | draft/validated/approved/rejected | GSI |
| `validation_notes` | String | Notes from validation | - |
| `related_signs` | List | Array of sign IDs | - |
| `confidence_score` | Number | AI confidence (0.0-1.0) | - |

### **Corpus Exports Table (`spokhand-corpus-exports`)**

| Field | Type | Description | Index |
|-------|------|-------------|-------|
| `id` | String | Unique export identifier | Primary Key |
| `corpus_id` | String | Reference to corpus | GSI |
| `export_format` | String | json/csv/txt | - |
| `status` | String | pending/processing/completed/failed | GSI |
| `created_by` | String | User ID who requested | GSI |
| `created_at` | String | ISO timestamp | GSI |
| `completed_at` | String | ISO timestamp | - |
| `download_url` | String | Download link | - |
| `error_message` | String | Error details if failed | - |

---

## 🔐 **Role-Based Access Control**

### **Permission Matrix**

| Role | Create Corpus | Edit Corpus | Delete Corpus | Add Segments | Edit Segments | Delete Segments | Export | View All |
|------|---------------|-------------|---------------|--------------|---------------|-----------------|---------|----------|
| **Translator** | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Segmenter** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Qualifier** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Expert** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Admin** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### **Access Control Implementation**

```python
# Example permission check in API
if (corpus.created_by != current_user['id'] and 
    'admin' not in current_user['roles'] and 
    'expert' not in current_user['roles']):
    return jsonify({'error': 'Insufficient permissions'}), 403
```

---

## 🚀 **API Endpoints**

### **Corpus Management**

#### **Create Corpus**
```http
POST /api/corpora
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "name": "Basic ASL Vocabulary",
  "description": "Essential signs for beginners",
  "language": "ASL",
  "metadata": {
    "difficulty_level": "beginner",
    "target_audience": "students"
  },
  "tags": ["beginner", "vocabulary", "education"]
}
```

#### **List Corpora**
```http
GET /api/corpora?status=active&language=ASL&user_id=user123
Authorization: Bearer <JWT_TOKEN>
```

#### **Get Corpus**
```http
GET /api/corpora/{corpus_id}
Authorization: Bearer <JWT_TOKEN>
```

#### **Update Corpus**
```http
PUT /api/corpora/{corpus_id}
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "name": "Updated Name",
  "status": "active",
  "tags": ["updated", "tags"]
}
```

#### **Delete Corpus**
```http
DELETE /api/corpora/{corpus_id}
Authorization: Bearer <JWT_TOKEN>
```

### **Text Segment Management**

#### **Add Text Segment**
```http
POST /api/corpora/{corpus_id}/segments
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "segment_type": "phrase",
  "metadata": {
    "category": "greetings",
    "difficulty": 1
  },
  "related_signs": ["sign_id_1", "sign_id_2"]
}
```

#### **List Segments**
```http
GET /api/corpora/{corpus_id}/segments?status=validated
Authorization: Bearer <JWT_TOKEN>
```

#### **Update Segment**
```http
PUT /api/segments/{segment_id}
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "text": "Updated text content",
  "status": "validated",
  "validation_notes": "Approved by expert"
}
```

#### **Delete Segment**
```http
DELETE /api/segments/{segment_id}
Authorization: Bearer <JWT_TOKEN>
```

### **Search & Export**

#### **Search Corpus**
```http
GET /api/corpora/{corpus_id}/search?q=hello&type=text
Authorization: Bearer <JWT_TOKEN>
```

#### **Export Corpus**
```http
POST /api/corpora/{corpus_id}/export
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "format": "json"
}
```

#### **Get Export Status**
```http
GET /api/corpora/exports/{export_id}
Authorization: Bearer <JWT_TOKEN>
```

#### **Download Export**
```http
GET /api/corpora/exports/{export_id}/download
Authorization: Bearer <JWT_TOKEN>
```

### **Statistics**

#### **Get Corpus Statistics**
```http
GET /api/corpora/{corpus_id}/stats
Authorization: Bearer <JWT_TOKEN>
```

---

## 🛠️ **Implementation Details**

### **Service Layer Architecture**

```python
class TextCorpusService:
    """Core business logic for text corpus management"""
    
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.corpora_table = self.dynamodb.Table('spokhand-text-corpora')
        self.segments_table = self.dynamodb.Table('spokhand-text-segments')
        self.exports_table = self.dynamodb.Table('spokhand-corpus-exports')
    
    def create_corpus(self, name, description, language, created_by, metadata=None, tags=None):
        """Create a new text corpus with validation"""
        
    def add_text_segment(self, corpus_id, text, segment_type, created_by, metadata=None):
        """Add text segment with automatic positioning"""
        
    def search_corpus(self, corpus_id, query, search_type='text'):
        """Search within corpus using multiple strategies"""
        
    def export_corpus(self, corpus_id, export_format, created_by):
        """Create export job with background processing"""
```

### **Key Features**

1. **Automatic Positioning**: Segments are automatically positioned sequentially
2. **Soft Deletes**: Data is marked as deleted rather than removed
3. **Background Processing**: Exports are processed asynchronously
4. **Flexible Metadata**: JSON-based metadata for extensibility
5. **Real-time Statistics**: Live calculation of corpus metrics

---

## 📱 **Frontend Integration**

### **React Components Structure**

```
src/components/
├── TextCorpusManager.tsx      # Main corpus management interface
├── CorpusEditor.tsx           # Create/edit corpus form
├── CorpusList.tsx             # List and filter corpora
├── TextSegmentForm.tsx        # Add/edit segment form
├── SegmentList.tsx            # Display segments in corpus
├── CorpusSearch.tsx           # Search interface
├── ExportManager.tsx          # Export creation and monitoring
└── CorpusStats.tsx            # Statistics dashboard
```

### **State Management**

```typescript
interface TextCorpusState {
  corpora: TextCorpus[];
  currentCorpus: TextCorpus | null;
  segments: TextSegment[];
  searchResults: TextSegment[];
  exports: CorpusExport[];
  loading: boolean;
  error: string | null;
}

const useTextCorpus = () => {
  const [state, setState] = useState<TextCorpusState>(initialState);
  
  const createCorpus = async (corpusData: CreateCorpusData) => {
    // Implementation
  };
  
  const addSegment = async (segmentData: CreateSegmentData) => {
    // Implementation
  };
  
  // ... other methods
};
```

---

## 🧪 **Testing Strategy**

### **Test Coverage**

- **Unit Tests**: Service layer methods with mocked dependencies
- **Integration Tests**: API endpoints with test database
- **Permission Tests**: Role-based access control validation
- **Edge Case Tests**: Error conditions and boundary cases

### **Running Tests**

```bash
# Run Epic 2 tests
cd src/
python test_text_corpus.py

# Run with coverage
python -m pytest test_text_corpus.py --cov=text_corpus_service --cov-report=html
```

---

## 🚀 **Deployment**

### **Environment Variables**

```bash
# Required
JWT_SECRET=your-super-secret-jwt-key
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Optional
DYNAMODB_TABLE_PREFIX=spokhand
PORT=5002
```

### **AWS Infrastructure**

```yaml
# CloudFormation template for Epic 2
Resources:
  TextCorporaTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: spokhand-text-corpora
      BillingMode: PAY_PER_REQUEST
      # ... table definition
      
  TextSegmentsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: spokhand-text-segments
      BillingMode: PAY_PER_REQUEST
      # ... table definition
      
  CorpusExportsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: spokhand-corpus-exports
      BillingMode: PAY_PER_REQUEST
      # ... table definition
```

---

## 📈 **Performance Considerations**

### **Optimization Strategies**

1. **Indexing**: Strategic GSI placement for common queries
2. **Pagination**: Large result sets are paginated
3. **Caching**: Redis caching for frequently accessed data
4. **Batch Operations**: Bulk operations for efficiency
5. **Async Processing**: Background jobs for heavy operations

### **Scalability**

- **Horizontal Scaling**: DynamoDB auto-scaling
- **Load Balancing**: Multiple API instances
- **CDN**: Static content delivery
- **Monitoring**: CloudWatch metrics and alarms

---

## 🔒 **Security Features**

### **Data Protection**

1. **JWT Authentication**: Secure token-based access
2. **Role-Based Access**: Granular permission control
3. **Input Validation**: Comprehensive data validation
4. **SQL Injection Protection**: Parameterized queries
5. **Audit Logging**: Complete action tracking

### **Compliance**

- **GDPR**: Data privacy and right to deletion
- **HIPAA**: Healthcare data protection (if applicable)
- **SOC 2**: Security and availability controls

---

## 🚀 **Getting Started**

### **Quick Start**

1. **Setup Database**
   ```bash
   cd src/
   python setup_database.py
   ```

2. **Start Services**
   ```bash
   # Terminal 1: Epic 1 (Authentication)
   python auth_api.py
   
   # Terminal 2: Epic 2 (Text Corpus)
   python text_corpus_api.py
   ```

3. **Run Demo**
   ```bash
   ./demo_epic2.sh
   ```

### **Sample Usage**

```python
from text_corpus_service import TextCorpusService

# Initialize service
service = TextCorpusService()

# Create corpus
corpus = service.create_corpus(
    name="My ASL Corpus",
    description="Personal collection",
    language="ASL",
    created_by="user@example.com"
)

# Add segments
segment = service.add_text_segment(
    corpus_id=corpus.id,
    text="Hello, how are you?",
    segment_type="phrase",
    created_by="user@example.com"
)

# Search corpus
results = service.search_corpus(
    corpus_id=corpus.id,
    query="hello",
    search_type="text"
)
```

---

## 🔮 **Future Enhancements**

### **Planned Features**

1. **Advanced Search**: Full-text search with Elasticsearch
2. **Version Control**: Git-like versioning for corpora
3. **Collaboration**: Multi-user editing with conflict resolution
4. **AI Integration**: Automatic text analysis and suggestions
5. **Real-time Updates**: WebSocket-based live collaboration

### **Integration Points**

- **Epic 3**: Enhanced Video Workspace
- **Epic 4**: AI Integration
- **Epic 5**: Lexicon Management
- **Epic 6**: Advanced Analytics

---

## 📚 **Additional Resources**

### **Documentation**

- [API Reference](./API_REFERENCE.md)
- [Database Schema](./DATABASE_SCHEMA.md)
- [Frontend Components](./FRONTEND_COMPONENTS.md)
- [Testing Guide](./TESTING_GUIDE.md)

### **Examples**

- [Sample Corpora](./examples/sample_corpora/)
- [API Usage Examples](./examples/api_usage/)
- [Frontend Integration](./examples/frontend_integration/)

---

## 🎯 **Success Metrics**

### **Performance Targets**

- **Response Time**: < 200ms for CRUD operations
- **Throughput**: 1000+ requests per second
- **Availability**: 99.9% uptime
- **Scalability**: Support 10,000+ corpora

### **User Experience**

- **Ease of Use**: 90% user satisfaction
- **Feature Adoption**: 80% of users use advanced features
- **Error Rate**: < 1% failed operations
- **Support Tickets**: < 5% of users need support

---

**Epic 2 is now complete and ready for Epic 3!** 🚀

The text corpus management system provides a solid foundation for organizing and managing sign language data, with comprehensive APIs, role-based access control, and scalable architecture that will support the advanced features planned for future epics.
