# CNN-in-Action_Fruits_Image_Classification

🍎 Fruit Classification using CNN with PyTorch
A deep learning project that classifies different types of fruits using Convolutional Neural Networks (CNN) with PyTorch and transfer learning. This project implements VGG19 architecture with custom classifier for accurate fruit recognition.
📋 Table of Contents

Overview
Dataset
Model Architecture
Installation
Usage
Results
Project Structure
Dependencies
Contributing

🎯 Overview
This project implements a fruit classification system using deep learning techniques. The model is built using PyTorch and employs transfer learning with VGG19 architecture to classify different types of fruits from images.
Key Features:

Transfer Learning: Utilizes pre-trained VGG19 model
Data Augmentation: Implements various image transformations for better generalization
Comprehensive Evaluation: Includes accuracy metrics, confusion matrix, and sample predictions
Visualization: Training/validation curves and prediction examples

📊 Dataset
The dataset contains images of 9 different fruit categories:

Apple fruit
Banana fruit
Cherry fruit
Chickoo fruit
Grapes fruit
Kiwi fruit
Mango fruit
Orange fruit
Strawberry fruit

Dataset Statistics:

Training Images: 80% of total dataset
Testing Images: 20% of total dataset
Image Size: Resized to 224x224 pixels
Data Augmentation: Random rotation, horizontal flip, center crop, normalization

🏗️ Model Architecture
Base Model: VGG19

Pre-trained: ImageNet weights
Feature Extraction: Frozen convolutional layers
Custom Classifier:

Linear layer: 25088 → 1024 features
ReLU activation
Dropout (0.4)
Linear layer: 1024 → 9 classes (number of fruit categories)
LogSoftmax activation



Training Configuration:

Optimizer: Adam
Loss Function: CrossEntropyLoss
Batch Size: 32 (training), 2 (testing)
Learning Rate: Default Adam parameters
Device: CUDA (if available)

🚀 Quick Start
Google Colab Setup (Recommended)

Open the notebook in Google Colab
Upload your fruit dataset ZIP file when prompted
Run all cells sequentially
View results including accuracy graphs and confusion matrix

Local Setup
bashgit clone https://github.com/yourusername/fruit-classification-cnn.git
cd fruit-classification-cnn
pip install -r requirements.txt
jupyter notebook fruit_classification.ipynb
💻 Usage
1. Prepare Dataset

Upload your fruit images dataset as a ZIP file
The code will automatically extract and organize the data
Ensure your dataset follows the standard folder structure:

dataset/
├── train/
│   ├── apple_fruit/
│   ├── banana_fruit/
│   └── ...
└── test/
    ├── apple_fruit/
    ├── banana_fruit/
    └── ...
2. Run Training
bashpython fruit_classification.py
Or use the Jupyter notebook:
bashjupyter notebook fruit_classification.ipynb
3. Key Functions:

Data Loading: Automatic dataset extraction and loading
Model Training: Transfer learning with VGG19
Evaluation: Accuracy calculation and confusion matrix
Prediction: Single image prediction with visualization

📈 Results & Performance
Model Achievements:

✅ High Accuracy: Achieved excellent classification performance across all fruit categories
✅ Robust Training: Smooth convergence without overfitting
✅ Comprehensive Evaluation: Detailed confusion matrix and per-class metrics
✅ Visual Predictions: Sample predictions with confidence scores

Key Performance Indicators:

Model Convergence: Loss decreases steadily across epochs
Accuracy Growth: Consistent improvement in both training and validation
Generalization: Strong performance on unseen test data
Class Balance: Good performance across all 9 fruit categories

Visualization Outputs:

Training/Validation loss and accuracy curves
Detailed confusion matrix heatmap
Sample prediction examples with confidence scores
Per-class performance breakdown

📁 Project Structure
fruit-classification-cnn/
├── README.md
├── requirements.txt
├── fruit_classification.ipynb    # Main notebook
├── fruit_classification.py       # Python script version
├── data/
│   └── unzipped_data/           # Extracted dataset
├── models/
│   └── vgg19_fruit_classifier.pth  # Saved model
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
└── docs/
    └── project_report.pdf       # Detailed report
📦 Dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.3.0
scikit-learn>=0.24.0
🔄 Training Process

Data Preprocessing: Images resized to 224x224, normalized using ImageNet statistics
Transfer Learning: VGG19 feature extractor frozen, custom classifier trained
Training Loop: Batch processing with gradient updates and loss tracking
Validation: Regular evaluation on test set with accuracy monitoring
Model Saving: Best performing model saved for inference

🎯 Key Features Implemented

✅ Transfer Learning with VGG19
✅ Data Augmentation for robust training
✅ Batch Processing for efficient training
✅ Real-time Monitoring of training progress
✅ Confusion Matrix for detailed evaluation
✅ Sample Predictions with confidence scores
✅ Model Persistence for deployment

🚧 Future Improvements

 Implement additional architectures (ResNet, EfficientNet)
 Add more fruit categories
 Deploy as web application
 Implement data augmentation strategies
 Add model ensemble techniques
 Create mobile app version

🤝 Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

PyTorch team for the excellent deep learning framework
VGG authors for the powerful architecture
Dataset contributors for fruit image collection
Open source community for various tools and libraries


Note: This project is developed for educational purposes as part of a machine learning course. The model achieves good performance on the given dataset and demonstrates practical implementation of transfer learning techniques.
