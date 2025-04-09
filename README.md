# 🌿 Crop Disease Detection using CNN

This project uses Convolutional Neural Networks (CNNs) to detect plant leaf diseases from images.

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Scikit-Learn

## 🗃️ Dataset
For demo purposes, dummy image data is generated. In real-world use, the [PlantVillage dataset](https://www.kaggle.com/emmarex/plantdisease) can be used for actual disease classification.

## 🏗️ Model Architecture
- Conv2D + MaxPooling2D layers
- Flatten + Dense layers
- Dropout for regularization
- Softmax output layer for multi-class classification

## 🧪 Output
The model achieves decent accuracy on a synthetic dataset. With real images, performance improves with preprocessing and data augmentation.

## ▶️ How to Run

```bash
pip install tensorflow scikit-learn numpy
python crop_disease_detector.py
