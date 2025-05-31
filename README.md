# Sign Language Recognition Platform

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
1. Ensure you have all dependencies installed (see requirements.txt).
2. From the project root, run:
   ```
   pytest src/tests/
   ```
   This will discover and run all test scripts in the `/tests` folder.

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
