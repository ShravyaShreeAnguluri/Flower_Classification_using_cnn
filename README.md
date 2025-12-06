# 🌸 Flower Classification using CNN

A deep learning project to classify flower images into different categories using **Convolutional Neural Networks (CNNs)**.  
This project demonstrates how computer vision can be applied to image classification tasks in a simple and reproducible way.

---

## 📌 Features
- Image preprocessing (resizing, normalization, augmentation).
- CNN-based model for multi-class flower classification.
- Training and validation workflow with accuracy/loss visualization.
- Prediction pipeline for new flower images.
- Easily extendable to other image datasets.

---

## 📊 Dataset
This project uses the **Flowers dataset**.  
If you don’t already have it, you can download similar datasets from Kaggle:  
👉 [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

⚠️ Note: The dataset is **not included** in this repository due to GitHub’s file size limits.  
You can download it directly in your notebook using the Kaggle API:

```bash
!pip install kaggle
!kaggle datasets download -d alxmamaev/flowers-recognition -p ./flowers
!unzip ./flowers/flowers-recognition.zip -d ./flowers
