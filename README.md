# Brain-Tumor-Classification
Brain Tumor Classification Using Deep Learning
SPIS Project 2025 Programmer(s): Shana Ibatuan

Brain tumors vary in size and location, making analysis challenging. MRI interpretation often requires a skilled neurosurgeon, which can be limited in developing countries. An automated cloud-based system can help generate reports quickly and accurately.

This project implements a deep learning model capable of classifying brain tumors from MRI images. The repository also includes an interactive web application where users can upload their own brain scan images to get a predicted tumor type, confidence score, and relevant information about the tumor.

The model was trained on the Brain Tumor MRI Dataset from Kaggle:
(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Images were preprocessed by resizing to 224x224 pixels and normalizing pixel values.

Features

Classifies brain tumors into multiple categories
Outputs predicted tumor type with confidence score
Provides educational information about the predicted tumor
Web interface for users to upload and analyze images interactively
Model

Architecture: CNN
Input shape: 224x224 RGB images
Output: Tumor class probabilities
Accuracy: 94% on test dataset
