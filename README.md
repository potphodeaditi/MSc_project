# Deep Learning for Respiratory Sound Classification and Continuous Monitoring

This repository contains the end-to-end machine learning pipeline, reproducibility framework, and deployment code developed for my MSc Artificial Intelligence thesis project at Nottingham Trent University (Graduated with High Commendation).

##  Project Overview
This project presents an automated AI solution for real-time respiratory sound monitoring and clinical classification. By analyzing acoustic bio-signals (coughing, wheezing, and breath anomalies), the system acts as an early warning and clinical decision support system to identify respiratory deterioration before crisis escalation.

##  Key Features
- **End-to-End Pipeline:** Automated pipeline handling raw audio processing, feature extraction (spectrogram transformation), and model evaluation.
- **Deep Learning Architectures:** Custom-optimized Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) hybrid architectures to extract both spatial acoustic patterns and temporal sequences from audio streams.
- **Deployment-Ready:** Features a functional `app.py` script for local web-app visualization and real-time inference deployment.
- **Reproducibility Focus:** Structured data logging, seed freezing, and optimized validation splits to guarantee workflow reproducibility across environments.

##  Repository Structure
- `project.ipynb`: Primary research workbook detailing data exploration, neural architecture design, and training iterations.
- `app.py`: Streamlit-based real-time audio analysis application script.
- `respiratory_cnn_best_tuned.h5`: Pre-trained, optimized weights for the best-performing deep neural network.
- `*.csv`: Standardized patient demographic profiles and consolidated audio metadata mappings.

## Core Technologies & Frameworks
- Python 3
- TensorFlow / Keras (Deep Learning Engineering)
- Librosa / Scikit-Learn (Audio Processing & Feature Engineering)
- Pandas / NumPy / Matplotlib (Data Informatics)
