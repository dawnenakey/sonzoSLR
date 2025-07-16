# SpokHand SLR - Sign Language Recognition System

A comprehensive sign language recognition system with advanced AI features, AWS cloud integration, and data management capabilities for American Sign Language (ASL) research and development.

## 🚀 New Features (Latest Update)

### Enhanced ASL-LEX Data Management
- **Custom Sign Type Management**: Add new sign type classifications with custom naming conventions
- **Video Preview**: Preview uploaded videos before submission with full video player controls
- **Batch Sign Type Updates**: Select multiple signs and update their sign types simultaneously
- **ASL-Specific Validation**: Comprehensive validation for ASL sign characteristics including:
  - Gloss format validation (UPPERCASE convention)
  - Handshape validation (A-Z handshapes)
  - Location validation (common ASL signing locations)
  - Sign type consistency checks
  - Movement pattern validation
  - Context-aware suggestions
- **Sign Type Filtering**: Filter signs by sign type in the interface
- **Advanced Analytics Dashboard**: 
  - Sign type distribution analysis
  - Validation status tracking
  - Confidence score analysis
  - Recent trends (last 30 days)
  - Custom type management
  - Examples and metadata per sign type

### Key Features

#### 🤖 Advanced AI Sign Recognition
- **Sign Spotting**: Automatically identify and localize individual signs within continuous signing
- **Context Disambiguation**: Use LLM reasoning to resolve ambiguous signs through linguistic context
- **I3D Spatiotemporal Features**: Advanced video feature extraction
- **Hand Shape Analysis**: ResNeXt-101-based hand shape recognition
- **Dictionary-based Matching**: Dynamic time warping for sign matching
- **LLM-powered Beam Search**: Context-aware disambiguation

#### 📊 Data Management System
- **Individual Video Upload**: Upload single videos with comprehensive metadata
- **Bulk Upload**: Process CSV/ZIP files with multiple videos and metadata
- **Real-time Progress Tracking**: Monitor upload and processing progress
- **Job Management**: Track and manage bulk upload jobs
- **Data Validation**: Comprehensive ASL-specific validation rules
- **Export Capabilities**: Export data in standardized formats

#### 🔧 Configuration Options
- **Feature Fusion Strategies**: Late Fusion (α = 0.9), Intermediate Fusion, Full Ensemble
- **Dictionary Size**: 1,000-4,373 signs
- **Beam Search Width**: 1-50
- **Fusion Weight (α)**: 0.0-1.0
- **Real-time Processing**: Live sign recognition

#### ☁️ AWS Cloud Integration
- **S3 Storage**: Secure video storage with automatic backup and versioning
- **DynamoDB**: NoSQL database for metadata and job tracking
- **Lambda Functions**: Serverless processing for video analysis
- **API Gateway**: RESTful API endpoints for data management
- **CloudFront**: Global content delivery for video streaming

## 🏗️ Architecture

### Backend Services
```
src/
├── asl_lex_service.py          # ASL-LEX data management service
├── app.py                      # Main Flask application
├── camera/                     # Camera integration
├── data/                       # Data processing utilities
├── inference.py                # AI inference engine
├── lambda_function.py          # AWS Lambda handler
└── utils/                      # Utility functions
```

### Frontend Components
```
frontend/src/
├── components/
│   ├── ASLLexDataManager.jsx   # Main data management interface
│   ├── VideoPlayer.jsx         # Video playback component
│   └── ui/                     # Reusable UI components
├── pages/                      # Application pages
└── api/                        # API client utilities
```

## 📋 Data Formats

### Individual Video Upload
```json
{
  "gloss": "HELLO",
  "english": "Hello",
  "sign_type": "isolated_sign",
  "handshape": "B",
  "location": "neutral space",
  "movement": "wave",
  "palm_orientation": "up",
  "dominant_hand": "right",
  "non_dominant_hand": "left",
  "frequency": 5,
  "age_of_acquisition": 2,
  "iconicity": 4,
  "lexical_class": "interjection",
  "tags": ["greeting", "basic"],
  "notes": "Common greeting sign",
  "confidence_score": 0.8
}
```

### Bulk Upload CSV Template
```csv
gloss,english,sign_type,handshape,location,movement,palm_orientation,dominant_hand,non_dominant_hand,frequency,age_of_acquisition,iconicity,lexical_class,tags,notes,confidence_score,video_filename
HELLO,Hello,isolated_sign,B,neutral space,wave,up,right,left,5,2,4,interjection,"greeting,basic",Common greeting sign,0.8,hello.mp4
THANK-YOU,Thank you,isolated_sign,B,chin,forward,up,right,left,5,2,3,interjection,"polite,basic",Polite expression,0.9,thank_you.mp4
```

### Sign Types
- **isolated_sign**: Single sign performed in isolation
- **continuous_signing**: Sign within continuous signing context
- **fingerspelling**: Manual alphabet or fingerspelling
- **classifier**: Classifier construction or classifier predicate
- **compound_sign**: Compound sign combining multiple elements
- **inflected_sign**: Sign with grammatical inflection
- **directional_sign**: Sign with directional movement
- **spatial_sign**: Sign involving spatial relationships
- **temporal_sign**: Sign indicating time or temporal aspect
- **manner_sign**: Sign indicating manner of action
- **number_sign**: Number or numeral sign
- **question_sign**: Question or interrogative sign
- **negation_sign**: Negative or negation sign
- **modality_sign**: Sign indicating modality or mood
- **other**: Other type of sign not listed above
- **Custom Types**: User-defined sign type classifications

## 🔍 ASL Validation Rules

### Gloss Validation
- ASL glosses should be written in UPPERCASE
- Minimum length of 2 characters
- Complex glosses with multiple hyphens may indicate compound signs

### Handshape Validation
- Valid handshapes: A, B, C, D, E, F, G, H, I, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- Invalid handshapes trigger validation errors

### Location Validation
- Common ASL locations: neutral space, head, face, chest, waist, chin, forehead, nose, mouth, ear, eye, cheek, shoulder, arm, hand, leg, foot
- Unusual locations generate warnings

### Sign Type Validation
- Fingerspelling signs should represent single letters or short words
- Number signs should have numeric glosses
- Question signs should contain question words (WHAT, WHERE, WHEN, WHO, WHY, HOW, WHICH)

### Movement Validation
- Common movements: wave, forward, backward, up, down, side, circle, straight, curved
- Suggestions for movement standardization

## 📊 Analytics Features

### Sign Type Distribution
- Total signs count
- Percentage breakdown by sign type
- Custom type management
- Validation status tracking

### Performance Metrics
- Average confidence scores per sign type
- Validation completion rates
- Recent upload trends (last 30 days)
- Examples and metadata per category

### Data Quality Insights
- Validation status distribution
- Confidence score analysis
- Upload frequency patterns
- Error and warning tracking

## 🚀 Deployment

### Prerequisites
- Python 3.8+
- Node.js 16+
- AWS CLI configured
- Docker (optional)

### Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Run the application
python src/app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### AWS Deployment
```bash
# Deploy to AWS
./deploy.sh

# Or use AWS CLI
aws cloudformation deploy --template-file infrastructure/aws-setup.yaml --stack-name spokhand-slr
```

## 🔧 Configuration

### Environment Variables
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Application Configuration
FLASK_ENV=production
API_BASE_URL=https://your-api-gateway-url.amazonaws.com
S3_BUCKET=your-s3-bucket-name
DYNAMODB_TABLE=your-dynamodb-table-name
```

### AI Model Configuration
```python
# Feature fusion weight
FUSION_ALPHA = 0.9

# Dictionary size
DICTIONARY_SIZE = 4373

# Beam search width
BEAM_WIDTH = 10

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7
```

## 📈 Usage Examples

### Adding Custom Sign Type
1. Click "Add Sign Type" button
2. Enter sign type name (lowercase with underscores)
3. Add description
4. Submit to create new classification

### Batch Sign Type Update
1. Select multiple signs using checkboxes
2. Click "Batch Update" button
3. Choose new sign type from dropdown
4. Confirm update for all selected signs

### Video Preview and Validation
1. Upload video file
2. Click "Preview Video" to review before submission
3. Click "Validate ASL Data" to check ASL-specific rules
4. Review validation results and suggestions
5. Submit when satisfied

### Analytics Dashboard
1. Click "Analytics" button
2. View sign type distribution
3. Check validation status
4. Analyze recent trends
5. Review performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `/docs` folder
- Review the AWS setup guides in the root directory

## 🔮 Roadmap

- [ ] Real-time sign language translation
- [ ] Mobile app for data collection
- [ ] Advanced gesture recognition
- [ ] Multi-language support (BSL, LSF, etc.)
- [ ] Integration with educational platforms
- [ ] Advanced analytics and reporting
- [ ] Machine learning model training interface
- [ ] Community-driven sign database
