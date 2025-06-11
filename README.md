# Sign Language Recognition Platform

[![Codecov](https://codecov.io/gh/dawnenakey/spokhandSLR/branch/main/graph/badge.svg)](https://codecov.io/gh/dawnenakey/spokhandSLR)
[![Python Tests](https://github.com/dawnenakey/spokhandSLR/actions/workflows/tests.yml/badge.svg)](https://github.com/dawnenakey/spokhandSLR/actions/workflows/tests.yml)
[![Lint](https://github.com/dawnenakey/spokhandSLR/actions/workflows/lint.yml/badge.svg)](https://github.com/dawnenakey/spokhandSLR/actions/workflows/lint.yml)

This project is a Streamlit web app for sign language recognition and model evaluation, built for the Masters of Data Science Capstone 2025 by Dawnena Key.

## Overview

This capstone demonstrates a video-based sign language recognition system using deep learning. The app is deployed on [Streamlit Cloud](https://streamlit.io/cloud) for interactive demonstration and also leverages AWS for scalable data storage and processing.

## Features

- Upload sign language videos for analysis and recognition
- Visualize model architecture, training metrics, confusion matrix, classification report, per-class accuracy, and ROC/AUC
- Real metrics from your trained model (classification report and ROC/AUC)
- Cloud-based, interactive demo for easy sharing and evaluation

## How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/dawnenakey/spokhandslr.git
   cd spokhandslr
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Cloud & AWS Setup

- The Streamlit app is deployed on [Streamlit Cloud](https://streamlit.io/cloud) for demonstration purposes.
- AWS is used for backend data storage and processing (already set up for this project).
- If you wish to reproduce the AWS setup, see `AWS_SETUP.md` and related scripts in this repo.
- **You do NOT need AWS to run the Streamlit demo locally or on Streamlit Cloud.**

## Files

- `streamlit_app.py`: Main Streamlit app
- `requirements.txt`: Python dependencies (minimal for Streamlit Cloud)
- `classification_report.csv`: Model evaluation metrics (replace with your own for custom results)
- `roc_results.pkl`: ROC/AUC data (replace with your own for custom results)
- `generate_roc_results.py`: Script to generate ROC data (optional, for reproducibility)
- `README.md`: This file
- `AWS_SETUP.md`, `aws_setup.sh`, `aws-policy.json`, etc.: AWS setup and deployment scripts (optional for demo)

## Notes

- If you want to use your own model results, replace `classification_report.csv` and `roc_results.pkl` with your own files.
- Sample data/videos are not included due to size; see the app for upload instructions.
- The app does **not** require local GPU or heavy ML dependencies for demonstration—everything is precomputed and visualized.

## Contact

Project by Dawnena Key, Masters of Data Science Capstone 2025  
For questions, please contact dawnena@icloud.com.

## How to Run Tests

### Automated Testing
The project uses GitHub Actions for automated testing and code quality checks. Every push and pull request triggers:
- Unit and integration tests
- Code coverage reporting
- Code linting and formatting checks

### Running Tests Locally
1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run all tests:
   ```bash
   pytest src/tests/
   ```

3. Run tests with coverage:
   ```bash
   pytest src/tests/ --cov=src --cov-report=term-missing
   ```

4. Run specific test categories:
   ```bash
   # Run only unit tests
   pytest src/tests/ -m "not integration"
   
   # Run only integration tests
   pytest src/tests/ -m "integration"
   ```

5. Run code quality checks:
   ```bash
   # Format code
   black .
   
   # Sort imports
   isort .
   
   # Check code style
   flake8 .
   
   # Type checking
   mypy src/
   ```

## Result Interpretation / Impact
- **Model Performance**: Model accuracy, loss, and other metrics are logged during training and validation. Results are interpreted in the context of real-world sign language recognition tasks, with a focus on both quantitative metrics and qualitative feedback from Deaf community members.
- **Impact**: The project aims to set a new standard for accessible, ethical, and community-driven sign language technology. By providing open, annotated datasets and robust ML models, Spokhand empowers researchers, developers, and accessibility advocates to build inclusive tools for global impact.

## Data Setup

### 1. Extract WLASL Data
The WLASL dataset is included as a submodule in this repository. To set up the data:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/dawnenakey/spokhandSLR.git
cd spokhandSLR

# If you already cloned without submodules, run:
git submodule update --init --recursive

# Create data directories
mkdir -p data/raw
mkdir -p data/processed

# Extract sample data (first 5 videos)
python WLASL/download_wlasl_samples.py
```

### 2. Data Structure
After extraction, your data directory will contain:
- `data/raw/`: Original video files and annotations
- `data/processed/`: Processed data ready for model training
- `data/collected/`: Your collected sign language data

### 3. Model Training
Once the data is extracted, you can train the model using:
```bash
python src/train.py
```

## License

This project is licensed under the terms of the MIT License.  
See the [LICENSE](LICENSE) file for details.
# trigger rebuild
