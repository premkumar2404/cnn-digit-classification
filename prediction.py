import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

# Load trained model
model = keras.models.load_model("model/mnist_cnn.h5")

# Ask for image path
file_path = input("Enter path of the digit image: ")

# Read & preprocess
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
if np.mean(img) > 127:
    img = 255 - img  # Invert colors if background is white
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img)
predicted_label = np.argmax(prediction)
print(f"Predicted Digit: {predicted_label}")

# Show image
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.show()
