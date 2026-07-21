# Action Recognition in Videos

## Project Overview

This project implements a deep learning-based Action Recognition system using PyTorch and the UCF101 dataset. The model classifies human actions from short video clips by learning both spatial and temporal features using a 3D Convolutional Neural Network (3D CNN).

---

## Dataset

- Dataset: UCF101
- Number of Classes: 101
- Dataset Type: Human Action Recognition Videos

---

## Technologies Used

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

## Project Structure

```
Action-Recognition-in-Videos/
│
├── Action_Recognition.ipynb
├── action_recognition_3dcnn.pth
└── README.md
```

---

## Solution Approach

The project follows the following pipeline:

1. Download and extract the UCF101 dataset.
2. Load the official train-test split.
3. Preprocess videos by:
   - Frame extraction
   - Temporal sampling
   - Frame resizing
   - Normalization
4. Create a custom PyTorch Dataset and DataLoader.
5. Build a 3D CNN model for action recognition.
6. Train the model using CrossEntropy Loss and the Adam optimizer.
7. Evaluate the model using:
   - Top-1 Accuracy
   - Top-5 Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
   - Inference Time
8. Visualize predictions on sample test videos.

---

## Model Architecture

The implemented model consists of:

- 3D Convolution Layers
- ReLU Activation
- Max Pooling
- Adaptive Average Pooling
- Fully Connected Classification Layer

The network learns both spatial and temporal features from video sequences.

---

## Setup Instructions

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Download and extract the UCF101 dataset.
4. Download and extract the official train-test split files.
5. Run all notebook cells in order.
6. Train the model or load the saved checkpoint.
7. Evaluate the model and visualize predictions.

---

## Dependencies

Install the required libraries:

```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-learn tqdm
```

---

## Results

The project evaluates the trained model using:

- Top-1 Accuracy
- Top-5 Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Inference Time per Video

---

## Files Included

- Action_Recognition.ipynb
- action_recognition_3dcnn.pth
- README.md

---

## Author

Nithya Stanly
