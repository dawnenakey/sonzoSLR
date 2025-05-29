# Spokhand Sign Language Recognition Platform

## Research Question & Challenge
Millions of Deaf and hard-of-hearing individuals face persistent barriers to communication due to the lack of accessible, real-time sign language translation tools. While recent advances in computer vision and deep learning have enabled progress in gesture and sign recognition, existing solutions remain limited in accuracy, scalability, and inclusivity—especially for large vocabularies and diverse signing styles.

**Research Question:**  
How can machine learning models be used to accurately and efficiently recognize American Sign Language (ASL) from video data, in order to improve accessibility and real-time communication for the Deaf and hard-of-hearing community?

**Challenge Statement:**  
Despite the availability of annotated sign language datasets, current ASL recognition systems struggle to generalize across signers, handle large vocabularies, and deliver real-time performance. This capstone project addresses these challenges by developing and evaluating a data-driven, video-based ASL recognition pipeline using the WLASL dataset for video modeling and ASL-LEX metadata for feature analysis and enrichment. The goal is to demonstrate a scalable, accurate, and inclusive approach to sign language recognition that can inform future accessibility technologies.

## Overview
Spokhand is building the world's largest annotated sign language database, leveraging AWS and AI/ML to develop an ethical, multilayered system for efficient sign language capture, categorization, and storage. Our platform combines advanced machine learning with community-driven annotation to create accurate, real-time sign language recognition.

## Dataset Description
- **ASL-LEX**: The primary dataset is ASL-LEX, a large-scale lexical database of American Sign Language. It includes features such as handshape, location, movement, and sign type, as well as metadata for each sign. Data is preprocessed and encoded for use in machine learning models.
- **WLASL**: The main video dataset for model training and evaluation. Videos and annotations are organized in `data/raw/`.
- **Data Structure**: Raw data is stored in `/data/raw`, processed data in `/data/processed`, and models in `/models`. Annotation and metadata are managed in `/src/data`.

## Differentiating Factors
This capstone project stands out from traditional large language models (LLMs) and other sign language recognition approaches in several key ways:

- **Video-Based Deep Learning:**  
  Unlike LLMs, which focus on text, this project uses a 3D CNN + LSTM architecture to process and recognize sign language directly from video data, capturing both spatial and temporal features.

- **Integration of Linguistic Metadata:**  
  By incorporating ASL-LEX metadata (handshape, movement, iconicity, etc.), the project enriches the feature space and enables deeper analysis of model performance across linguistic properties.

- **Custom Data Pipeline:**  
  The data loading and preprocessing pipeline is tailored for sign language videos, ensuring that each video is paired with its annotation and that the dataset is split for robust training and validation.

- **Transparent and Reproducible Workflow:**  
  The entire process—from data loading to model training, evaluation, and visualization—is documented and reproducible, with all code, requirements, and instructions provided in the repository.

- **Focus on Accessibility and Real-World Impact:**  
  The project is motivated by the real-world challenge of improving communication accessibility for the Deaf and hard-of-hearing community, with an emphasis on inclusivity and scalability.

## Data Loading and Usage
- **Dataset Structure:**  
  The project uses the WLASL dataset for video-based modeling and the ASL-LEX dataset for metadata and feature analysis. All video files and annotations are organized in the `data/raw/` directory.

- **Custom Dataset Class:**  
  A PyTorch `Dataset` class (`SignLanguageDataset`) is implemented to efficiently load video frames and corresponding annotations, enabling batch processing and integration with the training pipeline.

- **Feature Engineering:**  
  ASL-LEX features are joined with video labels where possible, allowing for enriched model input and more nuanced analysis of results.

- **Reproducibility:**  
  All data loading, preprocessing, and model training steps are documented and can be executed from scratch using the provided scripts and instructions.

## Evaluation
The performance of the sign language recognition model was evaluated using the following metrics and methods:

- **Accuracy:** The proportion of correctly predicted signs on the validation set.
- **Confusion Matrix:** Visualizes the distribution of correct and incorrect predictions across all sign classes.
- **Loss and Accuracy Curves:** Plots of training and validation loss/accuracy over epochs to assess learning progress and detect overfitting.
- **Feature Analysis:** Explored the relationship between ASL-LEX metadata (e.g., handshape, movement, iconicity) and model performance.
- **Error Analysis:** Reviewed misclassified examples to identify common challenges and potential areas for improvement.

All evaluation plots are saved in the `reports/` directory and are included in the project presentation.

## Results
- The model achieved an accuracy of **[insert your result]%** on the validation set.
- The confusion matrix and loss/accuracy curves demonstrate the model's strengths and areas for improvement.
- Feature analysis reveals that signs with higher frequency and more distinct handshapes are recognized more accurately.
- The project demonstrates the feasibility and impact of video-based ASL recognition using deep learning and linguistic metadata.

## Development Setup

### Prerequisites
- AWS Account with appropriate permissions
- Node.js 18+
- Docker
- Python 3.9+
- CUDA-capable GPU (for local ML development)

### Local Development
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure AWS credentials (if using S3 upload)
4. Start ML training: `python src/train.py`

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
   - (Optional) SageMaker deployment
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