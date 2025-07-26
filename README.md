# Image Classification of fruits :

## Problem Statement

In many real-world applications, recognizing objects from images is a critical task—especially in fields like agriculture, healthcare, retail, and surveillance. The goal of this project is to build a deep learning model that can classify fruit images into their correct categories. The challenge lies in dealing with variations in lighting, color, size, and shape of different fruits, all while ensuring the model generalizes well to unseen data.

## Objective

To develop a Convolutional Neural Network using PyTorch that classifies real-world fruit images with high accuracy, demonstrating the full pipeline from data preprocessing to evaluation.

## Introduction

Image classification is the task of assigning a label to an input image. With the rise of **deep learning**, CNNs have become the go-to architecture for image-related tasks due to their ability to automatically extract spatial features and patterns.

# This project involves:

1. Loading a real-world fruits dataset
   
2. Preprocessing and visualizing images
   
3. Building a CNN model from scratch
   
4. Training and evaluating the model
   
5. Displaying predictions and performance metrics

## What is a CNN?

A **Convolutional Neural Network** is a deep learning model designed for visual data. It mimics how humans recognize patterns like edges, colors, and textures.

## Key Components:

- **Convolutional Layers**: Extract local features using filters/kernels.
- **Activation Functions (ReLU)**: Add non-linearity.
- **Pooling Layers**: Downsample feature maps to reduce computation.
- **Fully Connected Layers**: Perform classification based on learned features.
- **Softmax Layer**: Outputs probabilities for each class.

## **Example**:  
If shown a picture of a banana, a CNN will detect yellow color, curved edges, and eventually classify it as “banana.”

## Tools & Libraries

- **Python**
- **PyTorch**
- **Torchvision**
- **Google Colab**
- **NumPy**
- **Matplotlib**
- **PIL (Python Imaging Library)**

## Dataset Description

- **Source**: [Kaggle - moltean/fruits](https://www.kaggle.com/moltean/fruits)
- **Type**: Image classification dataset with labeled fruit images.
- **Format**: Directory-structured with subfolders per class.
- **Images**: Thousands of fruit images across multiple categories.
- **Split**: `train/`, `test/`, and `validation/`

## Project Workflow

1. Import Libraries & Setup Environment
2. Load Dataset from Kaggle
3. Preprocess Images
   - Resize
   - Normalize
   - Convert to Tensors
4. Visualize Data Samples
5. Build CNN Model Using PyTorch
6. Train the Model
   - Define Loss Function (CrossEntropyLoss)
   - Use Optimizer (Adam)
   - Train for N epochs
7. Evaluate the Model
   - Accuracy
   - Confusion Matrix
   - Sample Predictions
8. Visualize Model Performance

## CNN Architecture

<img width="374" height="322" alt="image" src="https://github.com/user-attachments/assets/3883ace1-b9c7-41d9-a86e-d8e22c0c32b2" />

## Training Details

- **Epochs**: Customizable
- **Batch Size**: Typical values like 32 or 64
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Evaluation Metric**: Accuracy

## Challenges & Learnings

1. Understanding the effect of filter size, pooling, and stride
2. Importance of normalization and tensor formatting
3. Monitoring training loss to avoid overfitting
4. Improving results by tuning architecture and training parameters

## Future Work

- Add **Data Augmentation** for better generalization
- Use **Pre-trained Models** (e.g., ResNet, VGG)
- Deploy model using Flask or Streamlit
- Introduce **early stopping**, **learning rate scheduling**
- Use **transfer learning** for faster convergence

## How to Run

1. Upload the Jupyter Notebook (`.ipynb`) into **Google Colab**
2. Upload dataset ZIP or fetch from Kaggle using `kagglehub`
3. Run each cell step-by-step
4. Monitor training loss and accuracy
5. View final predictions and metrics

## Results

1. High accuracy on both validation and test sets
2. Model correctly identifies various fruit classes
3. Visualization of predictions shows strong generalization
4. Confusion matrix illustrates class-wise performance
