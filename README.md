# Image Classification of fruits :

## Aim :

The aim of this project is to develop a Convolutional Neural Network (CNN) model using PyTorch to perform image classification on a real-world dataset. The objective is to explore CNN architecture, data handling, and model evaluation techniques.

## Problem Statement :

With the increasing availability of image data across domains, it becomes crucial to automate the classification of images using deep learning. This project focuses on building an efficient CNN model to classify images into multiple fruit categories. It includes data loading, preprocessing, model design, training, and performance evaluation using visual metrics.

## Technologies & Libraries :

1. Python

2. PyTorch – for model building and training

3. Torchvision – for dataset and transformations

4. Matplotlib & Seaborn – for visualization

5. NumPy – numerical operations

6. Google Colab – for GPU-accelerated development

## What is CNN?

A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed for processing pixel data. CNNs use layers such as convolutional, pooling, and fully connected layers to automatically and adaptively learn spatial hierarchies of features from input images.

### Example:
A CNN learns to recognize a fruit by first identifying edges, then textures, and finally shapes.

## Why PyTorch?

1. PyTorch is a popular open-source deep learning library that provides:

2. Dynamic computation graph (eager execution)

3. GPU acceleration

4. Easy model building with torch.nn

5. Intuitive debugging and flexibility

## Dataset Details :

Source: Kaggle – moltean/fruits

Classes: Multiple fruit categories (apple, banana, etc.)

Format: Image files organized into class-specific folders

## Model Architecture Summary :

<img width="934" height="161" alt="image" src="https://github.com/user-attachments/assets/4ad178b5-0b18-40c6-a41c-6934041dacf8" />


<img width="763" height="628" alt="image" src="https://github.com/user-attachments/assets/0858bc4a-7341-44f2-a530-bce86cd4143d" />

## Project Workflow :

1. Data Preprocessing & Loading
Loaded dataset from Kaggle

Applied transforms (Resize, ToTensor, Normalize)

Split into train/test

2. Model Building
Defined CNN with 2 Conv+Pooling layers

Used ReLU and Softmax activations

Final dense layer outputs class probabilities

3. Training Loop
Tracked training & validation loss/accuracy

Used GPU (cuda) when available

Printed metrics per epoch

4. Evaluation & Prediction
Evaluated model on test set

Printed class-wise accuracy

Displayed confusion matrix & predictions

## Performance Graphs :


<img width="739" height="404" alt="image" src="https://github.com/user-attachments/assets/b7721afd-fd08-4066-af48-b2a9e55acba2" />


<img width="800" height="414" alt="image" src="https://github.com/user-attachments/assets/f652e8f5-4c7d-4dc9-861c-eaff66c83bca" />


<img width="712" height="410" alt="image" src="https://github.com/user-attachments/assets/da466426-ce2f-4d39-9fd5-118b80b51ff5" />


## Prediction and Output :


<img width="603" height="290" alt="image" src="https://github.com/user-attachments/assets/c95c8ff9-a60e-47f7-99ec-bd20bc86eb0c" />


<img width="976" height="665" alt="image" src="https://github.com/user-attachments/assets/9e9106a3-1d3f-466d-9452-313c7ba97d00" />


## Result :

The CNN model achieved high classification accuracy on the fruit dataset after training, with clear convergence in loss and accuracy curves. It effectively learned to distinguish between multiple image classes using PyTorch.




