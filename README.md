# 🧠 Handwritten Digit Recognition with Deep Learning (MNIST)

This project is a simple deep learning demo that uses the **MNIST dataset** to recognize handwritten digits (0–9). It uses **TensorFlow** and **Keras** to train a basic neural network and evaluate its performance.

## 🚀 Project Features
- Built with Python and TensorFlow
- Trained on MNIST dataset (60,000 images)
- Uses a simple feedforward neural network
- Predicts and visualizes handwritten digits

## 📊 Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## 📦 How to Run

### 1. Install dependencies:
```bash
pip install tensorflow matplotlib numpy
2. Run the project:
bash
Copy
Edit
python mnist_digit_classifier.py
3. Output:
Model trains on MNIST

Accuracy score is printed

Random test image is shown with model prediction

🧠 Model Summary
python
Copy
Edit
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
🎯 Sample Result
vbnet
Copy
Edit
Test doğruluğu: 0.97
Tahmin: 7 - Gerçek: 7
📁 Folder Structure
sql
Copy
Edit
mnist-digit-classifier/
├── mnist_digit_classifier.py
└── README.md
