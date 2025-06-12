# Deep Learning Based Pain Recognition
MTech Final Year Project - Deep Learning based Pain Recognition from Facial Expressions in Clinical Settings

This repository contains the implementation of lightweight deep learning models for automatic pain detection from facial videos, with a focus on edge-device feasibility. It includes models such as MobileNetV2-TCN and ResNet18-TCN, evaluated under LOSO validation, exploring a spatio-temporal approach for pain classifcation (no-pain vs highest level of pain).

## Dataset
This project uses the **BioVid Heat Pain Database** for model training and evaluation.  
**Note:** The dataset is not publicly available. Access must be requested from the Medical Psychology group at Ulm University.

## Features
- Video-based pain recognition using CNN-TCN architectures.
- Lightweight models suited for edge deployment.
- LOSO (Leave-One-Subject-Out) validation protocol.
- Performance metrics including accuracy, model parameters, and FLOPs.

## Input Modalities
- **Optical Flow Images (OFI):** Capture motion between consecutive video frames, highlighting temporal changes indicative of facial expressions related to pain.
- **Landmark Images:** Represent facial keypoints extracted from each frame, providing structural information useful for detecting subtle facial movements.

## System Overview
![corr drawio](https://github.com/user-attachments/assets/da47d65b-72da-465b-add2-1fefa3013958)

## Supervisor
This project is supervised by [Prof. Shivadarshan SL](https://erp.nitw.ac.in/ext/profile/cs-shivadarshan).
