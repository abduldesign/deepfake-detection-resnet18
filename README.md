Deepfake Detection Using ResNet-18

**Framework**

This project is implemented using PyTorch, a deep learning framework widely used for computer vision research and model development. The model was trained and evaluated in Google Colab using GPU acceleration when available.


**Dependencies and Libraries**

The following libraries were used in this project:

PyTorch

torchvision

KaggleHub

NumPy

Pillow (PIL)

Scikit-learn

Matplotlib

Seaborn

These libraries support dataset loading, training, evaluation, and result visualization.


**Dataset**

The dataset used is FaceForensics++ (C23 version) obtained from Kaggle:

Dataset link:
https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23


**Classes used:**

Original – real facial images

Deepfakes – manipulated images

The dataset is balanced with approximately the same number of real and fake images.
The C23 version includes medium compression, reflecting real-world conditions such as social media video uploads.


**Model Architecture**

The model used is ResNet-18, a convolutional neural network pre-trained on the ImageNet dataset (over 1 million images across 1,000 categories).


**Architecture components:**

Convolutional layers for feature extraction

Residual connections (skip connections) to improve gradient flow

ReLU activation function after each convolution

Global average pooling before classification

Fully connected layer modified to output 2 classes (Real vs Fake)

Training configuration:

Loss function: Cross-Entropy Loss

Optimizer: Adam

Technique: Fine-tuning (pre-trained model adapted to deepfake detection instead of training from scratch)

Experimental Results
Final Test Accuracy:

85.95%

**Confusion Matrix:**
Actual \ Predicted	Real	Fake
Real	839	161
Fake	120	880
Performance Metrics:
Metric	Value
Accuracy	85.95%
Precision (Real)	0.87
Recall (Real)	0.84
F1-Score (Real)	0.86
Precision (Fake)	0.85
Recall (Fake)	0.88
F1-Score (Fake)	0.86
Macro Avg F1	0.86
Weighted Avg F1	0.86
Interpretation

The model learns quickly during the first few epochs and then stabilizes.
Most samples are classified correctly, but some deepfakes are mistaken as real due to their high realism and compression effects.


**Selected Metrics and Limitations**

The model was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion matrix


**Other possible metrics that could also be useful:**

ROC-AUC

Equal Error Rate (EER)

Per-manipulation accuracy (by fake type)


**Research Paper Reference (ResNet-18-Based Study)**

The following paper is used as a reference related to ResNet-18-based deepfake detection:

Cited Paper:

Rafique, R., et al. (2023).
Deep fake detection and classification using error-level analysis and deep learning.
Scientific Reports, Nature.

https://doi.org/10.1038/s41598-023-34629-3
