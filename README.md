# Spokhand - Sign Language Recognition System

## Overview
Spokhand is a real-time sign language recognition system that uses computer vision and machine learning to interpret hand gestures. This project is designed to help bridge the communication gap between sign language users and non-signers.

## Features
- Real-time hand tracking using MediaPipe
- Gesture recognition for sign language
- Data collection and preprocessing utilities
- Model training pipeline
- User-friendly interface

## Project Structure
```
spokhandSLR/
├── src/
│   ├── models/         # Machine learning models
│   ├── utils/          # Utility functions
│   ├── data/           # Data collection and processing
│   └── tests/          # Test files
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/spokhandSLR.git
cd spokhandSLR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the main application:
```bash
python src/main.py
```

2. For data collection:
```bash
python src/data_collection.py
```

3. For model training:
```bash
python src/model_training.py
```

## Requirements
- Python 3.8+
- MediaPipe
- OpenCV
- NumPy
- TensorFlow
- Other dependencies listed in requirements.txt

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- MediaPipe for hand tracking capabilities
- OpenCV for computer vision support