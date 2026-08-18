# Deep Learning for Respiratory Sound Classification and Continuous Monitoring

**MSc Artificial Intelligence Research Project — Nottingham Trent University**

## Overview

This repository contains the research code, experimental work, trained deep learning model, and prototype application developed as part of my MSc Artificial Intelligence research project.

The project investigates the use of **deep learning and respiratory sound analysis** for automated respiratory sound classification and explores the potential use of respiratory acoustic signals for continuous monitoring.

The research combines:

- Audio signal processing
- Machine learning
- Deep learning
- Spectrogram-based feature representation
- CNN and CNN-LSTM architectures
- Model evaluation
- Prototype application development

## Abstract

Respiratory sounds contain acoustic patterns that may provide useful information about respiratory health.

This research investigates whether deep learning models can learn meaningful patterns from respiratory audio recordings and use these patterns for automated respiratory sound classification.

The project focuses on developing an end-to-end workflow covering:

1. Respiratory audio data preparation
2. Audio preprocessing
3. Feature extraction
4. Spectrogram generation
5. Deep learning model development
6. Model training
7. Model evaluation
8. Prototype application development

Two main deep learning approaches are investigated:

# Convolutional Neural Networks (CNN)
# CNN-LSTM architectures

The research also considers the challenges associated with respiratory audio data and the potential future application of acoustic analysis for continuous respiratory monitoring.

## Research Objectives

The main objectives of this research are:

1. To investigate respiratory sound as an acoustic source for respiratory condition classification.
2. To prepare and preprocess respiratory audio recordings for machine learning.
3. To extract meaningful acoustic representations from respiratory recordings.
4. To develop deep learning models for respiratory sound classification.
5. To investigate CNN-based architectures.
6. To investigate CNN-LSTM architectures for modelling temporal information.
7. To evaluate model performance using appropriate classification metrics.
8. To develop a prototype application for respiratory sound prediction.
9. To explore the potential of respiratory acoustic signals for continuous monitoring.


## Research Methodology

The overall research workflow is:


Respiratory Sound Data
        |
        v
        
Data Preparation
        |
        v
Data Cleaning

        |
        v
Audio Preprocessing

        |
        v
Feature Extraction


        |
        v
Spectrogram Representation

        |
        v
Deep Learning Models

        |
        +----------------+
        |                |
        v                v
       CNN           CNN-LSTM
        |                |
        +--------+-------+
                 |
                 v
          Model Training
          
                 |
                 v
          Model Evaluation
          
                 |
                 v
       Performance Analysis
       
                 |
                 v
       Prototype Application
       
       
## 1. Data Preparation

The research uses respiratory audio recordings together with associated metadata and diagnostic information.

The data preparation process includes:

Data organisation
Data cleaning
Metadata processing
Label preparation
Data quality checking

The original dataset is not redistributed in this repository.

## 2. Audio Preprocessing

Respiratory audio recordings are processed before being provided to the deep learning models.

The project uses Python-based audio processing techniques, including Librosa.

The processing workflow includes:

Audio loading
Resampling
Signal processing
Normalisation
Segmentation
Feature extraction

## 3. Feature Representation

The audio recordings are transformed into representations suitable for deep learning.

The project investigates spectrogram-based representations, allowing neural networks to learn acoustic patterns from respiratory recordings.

Spectrogram representations are particularly useful for applying image-based deep learning approaches such as CNNs to audio signals.

Deep Learning Models
Convolutional Neural Network (CNN)

CNN models are investigated to learn spatial and acoustic patterns from spectrogram representations of respiratory sounds.

The CNN approach is used to investigate whether learned acoustic representations can support automated respiratory sound classification.

## CNN-LSTM

The CNN-LSTM architecture combines:

CNN layers for acoustic feature extraction
LSTM layers for modelling sequential information

This approach allows both acoustic and temporal characteristics of respiratory sounds to be investigated.

The project also investigates bidirectional recurrent layers and dropout-based regularisation as part of the model development process.

## Hyperparameter Optimisation

The research includes experimentation with hyperparameter optimisation using Keras Tuner.

Hyperparameter experimentation is used to investigate model configurations and identify suitable settings for deep learning model development.
Model Evaluation

Model performance is evaluated using standard classification metrics, including:

Accuracy
Precision
Recall
F1-score
Confusion matrix

The final performance results should be interpreted based on the verified experimental results reported in the research notebook.

Experimental performance values are intentionally not listed here until they have been verified from the final research experiments.
Model Evaluation

Model performance is evaluated using standard classification metrics, including:

Accuracy
Precision
Recall
F1-score
Confusion matrix

The final performance results should be interpreted based on the verified experimental results reported in the research notebook.

Experimental performance values are intentionally not listed here until they have been verified from the final research experiments.

Audio Input

     |
     v
Audio Processing

     |
     v
Feature Extraction

     |
     v
Trained Deep Learning Model

     |
     v
Prediction

     |
     v
Result Display


## Dataset
Respiratory Sound Database

The research uses the Respiratory Sound Database, containing respiratory sound recordings collected from multiple patients together with associated metadata.

The dataset is used for research into respiratory sound classification and machine learning-based analysis.

The dataset is not redistributed in this repository.

Researchers wishing to reproduce the work should obtain the dataset from the authorised original source and follow its licence and usage requirements.

## Dataset Source

The dataset is available through Kaggle:

https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database


MSc_project/

|
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes

|
├── app/
|   └── app.py

|
├── models/
|   └── respiratory_cnn_best_tuned.h5

|
└── notebooks/
    └── project.ipynb

    
    
| File                                   | Description                                                                                          |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `notebooks/project.ipynb`              | Main research notebook containing data processing, experimentation, model development and evaluation |
| `app/app.py`                           | Streamlit prototype application                                                                      |
| `models/respiratory_cnn_best_tuned.h5` | Trained CNN model                                                                                    |
| `requirements.txt`                     | Python dependencies used by the project                                                              |
| `.gitignore`                           | Prevents selected files, including CSV datasets, from being uploaded                                 |
| `README.md`                            | Project documentation                                                                                |



| Technology / Library | Purpose                                 |
| -------------------- | --------------------------------------- |
| Python               | Research and development                |
| TensorFlow           | Deep learning                           |
| Keras                | Neural network development              |
| PyTorch              | Deep learning and audio experimentation |
| Torchaudio           | Audio processing                        |
| Librosa              | Audio signal processing                 |
| Noisereduce          | Audio noise reduction                   |
| Pandas               | Data processing                         |
| NumPy                | Numerical computation                   |
| Matplotlib           | Data visualisation                      |
| Seaborn              | Statistical visualisation               |
| Plotly               | Interactive visualisation               |
| Streamlit            | Prototype application                   |
| Keras Tuner          | Hyperparameter optimisation             |


##  Installation
# 1. Clone the repository
git clone https://github.com/potphodeaditi/MSc_project.git

Move into the project directory:

cd MSc_project

## 2. Install dependencies
pip install -r requirements.txt

For research experiments, a suitable Python environment such as a virtual environment or Conda environment is recommended.
Running the Research Notebook

The main research notebook is:

notebooks/project.ipynb

It can be opened using Jupyter Notebook, JupyterLab, or another compatible notebook environment.

For example:

jupyter notebook

Then open:

notebooks/project.ipynb

The notebook contains the main research workflow, including data processing, experimentation, model development and evaluation.
Running the Streamlit Application

The prototype application can be launched using:

streamlit run app/app.py

The trained model is located at:

models/respiratory_cnn_best_tuned.h5
Reproducibility

Reproducibility is an important consideration in this research project.

This repository provides:

Research notebook
Trained model
Application source code
Python dependency specification
Research documentation

The original respiratory dataset is not included in the repository.

To reproduce the experiments, researchers should:

Obtain the dataset from the authorised source.
Install the required Python dependencies.
Follow the preprocessing workflow described in the research notebook.
Run the experimental code.
Evaluate the trained models using the reported evaluation methodology.
Future improvements to reproducibility documentation may include:

Exact software versions
Dataset version information
Detailed preprocessing parameters
Feature extraction parameters
Model architecture specifications
Hyperparameter configurations
Training configuration
Random seeds
Evaluation methodology
Experimental results
Data and Research Ethics

Respiratory health-related data requires careful consideration of privacy, ethics and responsible research practices.

The original dataset is not redistributed through this repository.

Researchers using the dataset should:

Obtain the dataset from the authorised source.
Follow the dataset licence and terms of use.
Follow applicable research ethics requirements.
Avoid redistributing sensitive or personally identifiable information.
Use the data only for permitted research purposes.
Limitations

The research should be interpreted within the limitations of the available data and experimental setup.

Potential limitations include:

Dataset size and composition
Class distribution
Variability between respiratory recordings
Differences in recording environments
Differences between recording devices
Generalisation to unseen populations
Potential dataset bias
Differences between research datasets and real-world clinical environments

The developed models should therefore be considered research models and prototypes rather than clinically validated diagnostic systems.
Future Work

Potential future research directions include:

Evaluation using larger and more diverse respiratory sound datasets
Independent external validation
Improved model generalisation
Investigation of additional acoustic representations
Alternative deep learning architectures
Explainable AI methods
Real-time respiratory sound processing
Continuous respiratory monitoring
Robustness testing across recording devices
Further clinical validation
Investigation of multimodal respiratory monitoring

#### Results

The developed models were evaluated on a test set of **1,045 respiratory sound samples** using accuracy, precision, recall, F1-score and class-wise performance.

### Overall Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|---|---:|---:|---:|---:|
| **CNN** | **92.44%** | **51.64%** | **43.76%** | **47.00%** |
| **LSTM** | **88.33%** | **41.77%** | **24.18%** | **26.00%** |

The CNN achieved the highest overall test accuracy of **92.44%**, compared with **88.33%** for the LSTM model.

However, accuracy alone does not fully represent model performance because the dataset contains substantial class imbalance. Therefore, macro-averaged precision, recall and F1-score are also considered when interpreting the results.

### CNN Results

The final tuned CNN achieved:

- **Test accuracy:** 92.44%
- **Macro precision:** 51.64%
- **Macro recall:** 43.76%
- **Macro F1-score:** 47.00%
- **Weighted F1-score:** 92.00%

The CNN showed particularly strong performance for the majority COPD class, while performance varied considerably across minority classes.

### LSTM Results

The LSTM model achieved:

- **Test accuracy:** 88.33%
- **Macro precision:** 41.77%
- **Macro recall:** 24.18%
- **Macro F1-score:** 26.00%
- **Weighted F1-score:** 85.00%

The lower macro-averaged metrics indicate that the LSTM model had difficulty maintaining balanced performance across the different respiratory sound classes.

### Hyperparameter Optimisation

Hyperparameter optimisation was performed using **Keras Tuner**.

The best validation accuracy reported during tuning was approximately:

**90.91%**

The selected hyperparameters included:

| Hyperparameter | Selected Value |
|---|---:|
| Filters - Layer 1 | 32 |
| Filters - Layer 2 | 256 |
| Dense Units | 256 |
| Dropout | 0.4 |
| Learning Rate | 0.0001 |

The final tuned CNN training achieved a validation accuracy of approximately **92.82%** at the end of the reported training process.

### Class-wise Performance

The CNN classification report showed the following performance:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Class 0 | 0.00 | 0.00 | 0.00 |
| Class 1 | 0.86 | 0.60 | 0.71 |
| Class 2 | 0.57 | 0.57 | 0.57 |
| Class 3 (COPD) | 0.96 | 0.99 | 0.97 |
| Class 4 | 0.57 | 0.46 | 0.51 |
| Class 5 | 0.00 | 0.00 | 0.00 |
| Class 6 | 0.85 | 0.54 | 0.66 |
| Class 7 | 0.33 | 0.33 | 0.33 |

The results demonstrate strong classification performance for the majority COPD class but substantially lower performance for several minority classes.

### Interpretation

The results suggest that the CNN architecture performed better than the LSTM model on the reported test set.

However, the high overall accuracy is influenced by the strong representation of the COPD class in the test data. The macro-averaged metrics are substantially lower than the overall accuracy, highlighting the difficulty of achieving balanced performance across all respiratory conditions.

This class imbalance is an important limitation of the current experimental setup and should be considered when interpreting the model's generalisation capability.

Further investigation using class-balancing strategies, additional data, external validation and more diverse respiratory sound datasets would be valuable.

## Academic Context

This repository was developed as part of an MSc Artificial Intelligence research project at Nottingham Trent University.

The project combines:

Artificial intelligence
Machine learning
Deep learning
Audio signal processing
Respiratory sound analysis
Spectrogram-based representation
Time-series modelling
Application development

The research investigates the potential of respiratory acoustic analysis for automated classification and future continuous monitoring applications.
Author

Aditi Potphode

MSc Artificial Intelligence
Nottingham Trent University

GitHub: https://github.com/potphodeaditi

## Citation

If you use the research materials or code from this repository, please cite:

Potphode, A. (2026).
Deep Learning for Respiratory Sound Classification and Continuous Monitoring.
MSc Artificial Intelligence Research Project.
Nottingham Trent University.
Disclaimer

This project has been developed for academic and research purposes only.

The predictions produced by the models should not be interpreted as medical advice or a clinical diagnosis and should not replace assessment by a qualified healthcare professional.

The prototype application has not been clinically validated and should not be used to make medical decisions.

## Acknowledgements

I acknowledge the researchers and organisations responsible for the respiratory sound dataset and the open-source software libraries used in this project.

I also acknowledge the academic supervision and support provided during the MSc research project.
