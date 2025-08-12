🖋️ CNN Digit Classification

📌 Overview
This project implements a Convolutional Neural Network (CNN) to classify MNIST handwritten digits (0–9).
It demonstrates:

Data preprocessing (normalization, reshaping, one-hot encoding)
CNN architecture with convolution, pooling, and dense layers
Model training & evaluation
Prediction from user-provided images using OpenCV

🛠️ Tech Stack
Python 🐍
TensorFlow / Keras – Deep Learning framework
OpenCV – Image preprocessing
NumPy – Numerical operations
Matplotlib – Visualization

📂 Repository Structure

├── model/                   # Pre-trained model (mnist_cnn.h5)
├── Sample Images/           # Test images for prediction
├── train.py                 # Script to train the CNN model
├── prediction.py            # Script to predict digit from image
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── LICENSE                  # License file


🚀 How to Run
1️⃣ Clone the Repository

git clone https://github.com/yourusername/cnn-digit-classification.git
cd cnn-digit-classification

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (Optional)
python train.py
💡 This will save mnist_cnn.h5 inside the model/ folder.

4️⃣ Predict from an Image
python prediction.py
Enter the path of a 28x28 grayscale image.

The model will predict the digit and display it.

📸 Example Prediction
Below is an example of the CNN predicting a handwritten digit:


📊 Model Architecture

Input Layer: 28x28x1 (grayscale image)
Conv2D: 32 filters, kernel size (3x3), activation='relu'
MaxPooling2D: pool size (2x2)
Flatten
Dense: 64 neurons, activation='relu'
Dense: 10 neurons, activation='sigmoid'

📥 Pre-trained Model
Download the trained model: mnist_cnn.h5
(Place in model/ folder before running prediction.py)

👨‍💻 Author
Prem Kumar
📧 Email: prem598826@gmail.com
🌐 GitHub: premkumar2404

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

