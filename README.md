# cnn-digit-classification
Handwritten digit classification using Convolutional Neural Networks (CNN) with Python, TensorFlow/Keras, and OpenCV.

# Handwritten Digit Classification using CNN

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify MNIST handwritten digits (0–9).  
It includes:
- Data preprocessing
- Model training and evaluation
- Real-time image prediction using OpenCV

---

## 🚀 Features
- Achieves ~98% accuracy on MNIST test data
- Custom CNN architecture with Conv2D, MaxPooling2D, Flatten, Dense layers
- Accepts user-provided digit images for prediction
- Visualizes predictions and model performance

---

## 🛠 Tech Stack
- **Python 3**
- **TensorFlow/Keras**
- **NumPy**
- **Matplotlib**
- **OpenCV**

---

## 📊 Model Architecture
1. **Conv2D** – 32 filters, 3×3 kernel, ReLU activation  
2. **MaxPooling2D** – 2×2 pool size  
3. **Flatten** – converts 3D feature maps to 1D  
4. **Dense** – 64 neurons, ReLU activation  
5. **Dense** – 10 neurons, Softmax activation  

---

## 📂 Dataset
The project uses the **MNIST dataset** from Keras:
```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## How to Run

### 1. Clone the repository
git clone https://github.com/yourusername/cnn-digit-classification.git
cd cnn-digit-classification

### 2. Install dependencies
pip install -r requirements.txt

### 3. Train the model (optional)
python train.py

### 4. Predict a digit from an image
python predict.py

