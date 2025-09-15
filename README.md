# CleanSort AI - Waste Classification System

## Overview
CleanSort AI is an intelligent waste classification system that uses deep learning to help users identify and properly sort different types of waste materials. The system can classify waste into four categories: metal, organic, paper, and plastic.

## Features
- Real-time waste classification through webcam
- User-friendly GUI interface for image upload and classification
- High-accuracy deep learning model
- Detailed recycling instructions for each waste type
- Visual confidence indicators for predictions
- Training performance visualization

## Project Structure
```
CleanSort_AI/
│
├── dataset/                # Training and testing datasets
│   ├── train/             # Training images
│   └── test/              # Testing images
│
├── models/                # Trained model storage
│   └── waste_classifier.h5
│
├── output/                # Training visualization output
│   └── training_performance.png
│
├── src/                   # Source code
│   ├── train_model.py     # Model training script
│   ├── classify_waste.py  # Real-time classification script
│   └── gui_classifier.py  # GUI application
│
├── requirements-dev.txt   # Development dependencies
└── requirements-pi.txt    # Raspberry Pi dependencies
```

## Model Architecture
- Input Size: 224x224 pixels
- Convolutional Neural Network (CNN)
- Training Parameters:
  - Batch Size: 32
  - Epochs: 15
  - Optimizer: Adam
  - Loss Function: Categorical Crossentropy

## Dataset
The model is trained on a dataset containing four categories of waste:
- Metal
- Organic
- Paper
- Plastic

Each category contains multiple images showing different variations and angles of waste materials.

## Performance
The model's performance can be visualized through training metrics saved in `output/training_performance.png`, showing:
- Training and Validation Accuracy
- Training and Validation Loss
