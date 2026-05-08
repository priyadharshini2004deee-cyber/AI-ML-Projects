# 🩺 Multi-class COVID-19 Detection from Chest X-ray Images

## 📌 Project Overview

This project focuses on building a deep learning-based medical imaging system capable of classifying Chest X-ray images into three categories:

- COVID-19
- Viral Pneumonia
- Normal

The system uses Transfer Learning, Convolutional Neural Networks (CNNs), and Explainable AI techniques such as Grad-CAM to provide interpretable predictions through a Streamlit web application.

---

## 🚀 Skills Gained

- Medical Image Preprocessing & Augmentation
- Deep Learning Architectures (CNNs)
- Transfer Learning using MobileNetV2
- Multi-class Image Classification
- Model Evaluation & Fine-Tuning
- Explainable AI using Grad-CAM
- Streamlit Deployment
- End-to-End AI Workflow

---

## 🏥 Domain

Healthcare, Medical Imaging, Artificial Intelligence for Diagnostics

---

## ❗ Problem Statement

Rapid and accurate diagnosis of COVID-19 from chest X-ray images can significantly improve patient outcomes and reduce pressure on healthcare systems.

The objective of this project is to build a multi-class classification model capable of distinguishing between:

- COVID-19
- Viral Pneumonia
- Normal Chest X-ray images

using deep learning techniques.

---

## 💼 Business Use Cases

### Clinical Support
Assist radiologists with quick patient triage in hospitals.

### Remote Healthcare
Provide diagnostic support in low-resource or remote areas.

### Public Health Screening
Enable automated large-scale screening systems.

### Educational Purpose
Useful for medical learning and AI-based healthcare research.

---

## 📂 Dataset Information

### Dataset
COVID-19 Image Dataset by Pranav Raikokte from Kaggle

### Dataset Classes

- COVID
- Viral Pneumonia
- Normal

### Dataset Structure

```text
train/
test/
```

---

## 🔍 Data Preprocessing

- Image resizing to 128 × 128
- Pixel normalization
- Data augmentation:
  - Rotation
  - Horizontal Flip
  - Brightness Adjustment
  - Zoom Augmentation

---

## 🧠 Model Development

### Baseline Model

- Custom CNN architecture

### Advanced Model

- Transfer Learning using MobileNetV2
- Fine-tuned classification layers

### Frameworks Used

- TensorFlow
- Keras

---

## 📊 Model Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- Grad-CAM Explainability

---

## 🔥 Explainable AI: Grad-CAM

Grad-CAM visualization is implemented to highlight important lung regions used by the model during prediction.

This improves:

- Transparency
- Clinical Trust
- Model Interpretability

---

## 🌐 Streamlit Deployment

The project includes a fully functional Streamlit web application with:

- Upload Chest X-ray Image
- AI Prediction
- Confidence Score
- Grad-CAM Visualization
- PDF Report Download
- Professional UI

---

## 📁 Project Structure

```text
COVID-19-Xray-AI-Detector/
│
├── archive/
├── covid.ipynb
├── covidapp_all.py
├── final_model.h5
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```bash
streamlit run covidapp_all.py
```

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- PIL
- ReportLab

---

## 📈 Expected Results

- Accurate multi-class chest X-ray classification
- Visual explanation using Grad-CAM
- User-friendly AI medical diagnosis interface
- Explainable AI workflow for healthcare applications

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only.

It is NOT intended for real-world medical diagnosis or clinical use.

---

## 👩‍💻 Developed By

Priyadharshini

AI & Machine Learning Project Portfolio