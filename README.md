# Spokhand Sign Language Recognition Platform

## Problem Definition / Research Question
Millions of Deaf and hard-of-hearing individuals face barriers to communication due to the lack of accessible, real-time sign language translation tools. Spokhand aims to address this gap by building the world's largest annotated sign language database and developing AI/ML-powered systems for efficient, ethical, and scalable sign language recognition. The core research question: How can we leverage multimodal data and machine learning to create accurate, inclusive, and real-time sign language recognition systems that serve diverse signing communities?

## Overview
Spokhand is building the world's largest annotated sign language database, leveraging AWS and AI/ML to develop an ethical, multilayered system for efficient sign language capture, categorization, and storage. Our platform combines advanced machine learning with community-driven annotation to create accurate, real-time sign language recognition.

## Dataset Description
- **ASL-LEX**: The primary dataset is ASL-LEX, a large-scale lexical database of American Sign Language. It includes features such as handshape, location, movement, and sign type, as well as metadata for each sign. Data is preprocessed and encoded for use in machine learning models. Additional video data is collected using OAK-D cameras and annotated for training and evaluation.
- **Data Structure**: Raw data is stored in `/data/raw`, processed data in `/data/processed`, and models in `/models`. Annotation and metadata are managed in `/src/data`.

## Architecture
- **Data Collection**: OAK-D camera integration for 3D motion capture
- **Storage**: Amazon S3 for secure video and annotation storage
- **Processing**: AWS Lambda for automated video processing
- **ML Pipeline**: Amazon SageMaker for model training and inference
- **Annotation Platform**: Custom web interface for linguistic annotation
- **Real-time Inference**: API Gateway and Lambda for low-latency predictions

## Components

### 1. Data Collection Module
- OAK-D camera integration
- Real-time 3D motion capture
- Depth sensing and hand tracking
- Secure upload to S3

### 2. Machine Learning Pipeline
- Real-time sign language recognition
- Hand pose estimation
- Gesture classification
- Model training and deployment
- Continuous learning from annotations

### 3. Annotation Platform
- Video segmentation
- Gloss tagging
- Semantic labeling
- Community review system
- Quality assurance workflows

### 4. AWS Infrastructure
- Serverless architecture
- Secure data pipeline
- Scalable storage
- ML model deployment
- Real-time inference API

## Development Setup

### Prerequisites
- AWS Account with appropriate permissions
- OAK-D camera
- Node.js 18+
- Docker
- Python 3.9+
- CUDA-capable GPU (for local ML development)

### Local Development
1. Clone the repository
2. Install dependencies: `npm install` and `pip install -r requirements.txt`
3. Configure AWS credentials
4. Start the development server: `npm run dev`
5. Start ML training: `python src/train.py`

### AWS Deployment
1. Set up AWS infrastructure using provided CloudFormation templates
2. Configure environment variables
3. Deploy using AWS CDK
4. Start ML training job: `python scripts/start_training.py`

## ML Model Training
1. Data Preparation
   - Video preprocessing
   - Hand detection and tracking
   - Feature extraction
   - Dataset creation

2. Model Development
   - Architecture selection
   - Training pipeline
   - Validation metrics
   - Model optimization

3. Deployment
   - Model packaging
   - SageMaker deployment
   - API integration
   - Performance monitoring

## Community Guidelines
- All development must prioritize Deaf community needs
- Maintain linguistic integrity in annotations
- Follow accessibility-first design principles
- Ensure ethical data collection and usage
- Open collaboration on ML models

## License
Proprietary - Spokhand 501(c)(3)

## Contact
For more information about Spokhand and our mission, visit [spokhand.org](https://spokhand.org)

## How to Run Tests
1. Ensure you have all dependencies installed (see requirements.txt).
2. From the project root, run:
   ```
   pytest src/tests/
   ```
   This will discover and run all test scripts in the `/tests` folder.

## Result Interpretation / Impact
- **Model Performance**: Model accuracy, loss, and other metrics are logged during training and validation. Results are interpreted in the context of real-world sign language recognition tasks, with a focus on both quantitative metrics and qualitative feedback from Deaf community members.
- **Impact**: The project aims to set a new standard for accessible, ethical, and community-driven sign language technology. By providing open, annotated datasets and robust ML models, Spokhand empowers researchers, developers, and accessibility advocates to build inclusive tools for global impact.