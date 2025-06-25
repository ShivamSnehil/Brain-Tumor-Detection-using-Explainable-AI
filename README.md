
# 🧠 Brain Tumor Detection from MRI Scans using CNNs and Explainable AI

This project focuses on developing a deep learning pipeline for accurately detecting and classifying brain tumors from MRI scans using Convolutional Neural Networks (CNNs). The objective was not just high accuracy but also explainability, ensuring the model’s decisions could be interpreted by visualizing which parts of the brain scan contributed to the prediction using Grad-CAM.

---

## 🎯 Problem Statement

Brain tumors are a critical health concern, and timely diagnosis is vital. Radiologists often rely on MRI scans to detect and classify tumors, but manual diagnosis can be time-consuming and error-prone. This project aims to automate and enhance the diagnostic process by:

- Classifying MRI images into four categories:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- Providing visual explanations for each classification to make the model interpretable.

---

## 📚 Dataset

🔗 [Dataset Link ](https://drive.google.com/drive/folders/1B3gkttpW2KW_4UByrV9b_vlfAPnloyFf?usp=sharing)

- Over 7,000 images categorized into 4 classes.
- Organized into **Training** and **Testing** folders, each with subfolders for the 4 tumor types.
- Image format: `.jpg` files of varying dimensions.
```
summmer dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```
---

## 🧠 Methodology

### 1. Data Preprocessing
- Images were resized to 224x224 pixels.
- Image normalization was done using ImageNet mean and standard deviation.
- Data augmentation included:
  - Random horizontal flipping
  - Standardization

This helps in improving model generalization and handling class imbalance.

### 2. Model Architectures

Several models were tested and compared:

#### 🔹 Custom CNN
- A simple handcrafted CNN.
- **Test accuracy:** <60%
- Served as a baseline model.

#### 🔹 Pretrained AlexNet
- Pretrained on ImageNet and fine-tuned on the MRI dataset.
- **Test accuracy:** >85%

#### 🔹 VGG16 + Hypercolumn Technique
- Used intermediate layers (conv1 to conv5) and upsampled them to form a hypercolumn.
- Helped incorporate multi-level feature representation.
- **Test accuracy:** ~82%

#### 🔹 Final Model — Baseline VGG16
- Pretrained VGG16 with fully connected layers fine-tuned.
- No hypercolumn; only final convolutional features used.
- Achieved the best performance with **96% test accuracy**.

### 3. Explainability with Grad-CAM

To interpret model predictions, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was used.

- Grad-CAM overlays a heatmap on the original image, showing which areas the model focused on to make its prediction.
- This enhances transparency and trustworthiness, crucial for medical applications.

**Example:** If the model predicts “Pituitary Tumor,” the Grad-CAM image will highlight the specific region in the MRI that led to that prediction.

---

## 🛠 Technologies Used

### Programming Language:
- Python

### Libraries & Frameworks:
- PyTorch
- Torchvision
- pytorch-grad-cam

### Tools:
- Google Colab (for training with GPU)
---

## 📁 Repository Structure

- `code.ipynb` – Full pipeline: preprocessing, training, evaluation, and Grad-CAM
- `Sample images of dataset/` – Sample input images from dataset
- `Output samples/` – Grad-CAM output visualizations

---

## 🚀 How to Run

1. Download the dataset from the Kaggle link.
2. Place the dataset in your Google Drive in the following structure:


