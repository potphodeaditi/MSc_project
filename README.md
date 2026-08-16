# Deep Learning for Respiratory Sound Classification and Continuous Monitoring

**MSc Artificial Intelligence Research Project — Nottingham Trent University**

This repository contains the research code, experimental work, trained deep learning model, and prototype application developed as part of my MSc Artificial Intelligence research project.

The project investigates the use of **deep learning and respiratory sound analysis** for automated respiratory sound classification and the potential application of acoustic signals for continuous respiratory monitoring.

## Abstract

Respiratory sounds contain acoustic patterns that may provide useful information about respiratory health.

This project investigates whether deep learning techniques can learn meaningful patterns from respiratory audio recordings and use these patterns for automated classification.

The research focuses on audio preprocessing, feature extraction, deep learning model development, model evaluation, and the implementation of a prototype application for respiratory sound analysis.

Two main deep learning approaches are investigated:

* **Convolutional Neural Networks (CNN)**
* **CNN-LSTM architectures**

The project aims to explore the potential of these approaches for respiratory sound classification while considering the challenges associated with real-world respiratory audio data.


## Research Objectives

The main objectives of this research are:

1. To investigate respiratory sound as an acoustic source for respiratory condition classification.
2. To preprocess and prepare respiratory audio data for machine learning.
3. To extract meaningful acoustic representations from respiratory recordings.
4. To develop deep learning models for respiratory sound classification.
5. To investigate CNN and CNN-LSTM architectures.
6. To evaluate model performance using appropriate classification metrics.
7. To develop a prototype application for respiratory sound prediction.
8. To investigate the potential use of respiratory acoustic signals for continuous monitoring.


## Research Methodology

The overall research workflow is:

Respiratory Sound Data
        │
        ▼
Data Preparation
        │
        ▼
Data Cleaning
        │
        ▼
Audio Preprocessing
        │
        ▼
Feature Extraction
        │
        ▼
Audio / Spectrogram Representation
        │
        ▼
Deep Learning Models
        │
        ├──────────────┐
        ▼              ▼
       CNN         CNN-LSTM
        │              │
        └──────┬───────┘
               ▼
        Model Training
               │
               ▼
        Model Evaluation
               │
               ▼
       Performance Analysis
               │
               ▼
       Prototype Application


### Data Preparation

The project uses respiratory audio recordings together with associated metadata and diagnostic information.

The data preparation process includes:

* Data organisation
* Data cleaning
* Metadata processing
* Label preparation
* Data quality checking

### Audio Processing

Respiratory audio recordings are processed before being used by the deep learning models.

The project uses Python-based audio processing techniques, including **Librosa**, for working with respiratory sound signals.

The processing workflow includes techniques such as:

* Audio loading
* Resampling
* Signal processing
* Normalisation
* Segmentation
* Feature extraction

### Feature Representation

Audio signals are transformed into representations suitable for deep learning.

The project investigates spectrogram-based representations to allow neural networks to learn acoustic patterns from respiratory recordings.

### Deep Learning Models

#### Convolutional Neural Network (CNN)

CNN models are investigated to learn relevant patterns from audio representations such as spectrograms.

#### CNN-LSTM

A CNN-LSTM architecture combines:

* **CNN layers** for acoustic feature extraction
* **LSTM layers** for modelling sequential information

This approach allows both acoustic and temporal characteristics of respiratory sounds to be investigated.


## Model Evaluation

The models are evaluated using standard classification metrics.

The evaluation process considers metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Final performance values will be reported based on the verified experimental results from the research notebook.


## Prototype Application

The repository includes a Streamlit-based application implemented in `app.py`.

The application provides a prototype interface for demonstrating respiratory sound analysis using the trained model.

The general workflow is:


Audio Input
     │
     ▼
Audio Processing
     │
     ▼
Feature Extraction
     │
     ▼
Trained Deep Learning Model
     │
     ▼
Prediction
     │
     ▼
Result Display


The application is intended for **academic and research demonstration purposes**.

It should not be considered a clinically validated diagnostic system.


## Repository Contents

The current repository contains the following main files:

| File                                      | Description                                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `project.ipynb`                           | Main research notebook containing data analysis, model development, training and experimentation |
| `app.py`                                  | Streamlit application for demonstrating the trained model                                        |
| `respiratory_cnn_best_tuned.h5`           | Trained CNN model                                                                                |
| `patient_diagnosis.csv`                   | Patient diagnosis dataset                                                                        |
| `cleaned_patient_diagnosis.csv`           | Processed diagnosis data                                                                         |
| `ccleaned_patient_diagnosis1.csv`         | Additional processed diagnosis data                                                              |
| `combined_audio_metadata.csv`             | Combined respiratory audio metadata                                                              |
| `combined_audio_metadata_with_labels.csv` | Audio metadata with associated labels                                                            |
| `.gitignore`                              | Git configuration for excluding selected files                                                   |
| `.gitattributes`                          | Git repository attributes                                                                        |


## Technologies

The project was developed using the following technologies and libraries:

| Technology   | Purpose                               |
| ------------ | ------------------------------------- |
| Python       | Research and development              |
| TensorFlow   | Deep learning                         |
| Keras        | Neural network development            |
| Librosa      | Audio processing                      |
| Scikit-learn | Machine learning and model evaluation |
| Pandas       | Data processing                       |
| NumPy        | Numerical computation                 |
| Matplotlib   | Data visualisation                    |
| Streamlit    | Prototype web application             |



## Getting Started

### Requirements

To run the project, you may need:

* Python 3.x
* Jupyter Notebook
* TensorFlow
* Keras
* Librosa
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Streamlit

### Clone the Repository

```bash
git clone https://github.com/potphodeaditi/MSc_project.git
cd MSc_project
```

### Run the Research Notebook

Open the project notebook using Jupyter:

```bash
jupyter notebook
```

Then open:

```text
project.ipynb
```

The notebook contains the main research workflow and experimental development.

### Run the Streamlit Application

The prototype application can be launched using:

```bash
streamlit run app.py
```

## Reproducibility

Reproducibility is an important consideration in this research project.

The repository contains the main research notebook, trained model, data-processing files, and application code used during development.

Future improvements to the repository will include more detailed documentation of:

* Software versions
* Data preprocessing
* Feature extraction
* Model architecture
* Hyperparameters
* Training configuration
* Evaluation methodology
* Experimental results

## Data and Research Ethics

This project involves respiratory health-related information and therefore requires careful consideration of data privacy and research ethics.

The datasets included in this repository should only be used in accordance with the relevant dataset licence, research permissions, and applicable data governance requirements.

Sensitive or personally identifiable information should not be publicly distributed.

Where datasets are subject to access restrictions, researchers should obtain the data from the original authorised source and follow its terms of use.


## Limitations

This research should be interpreted within the limitations of the available data and experimental setup.

Potential limitations include:

* Dataset size and composition
* Class distribution
* Variability between respiratory recordings
* Differences in recording environments and devices
* Generalisation to unseen populations
* Differences between research datasets and real-world clinical environments

The models developed in this project should therefore be considered **research models and prototypes rather than clinically validated diagnostic systems**.

## Future Work

Future research could investigate:

* Larger and more diverse respiratory sound datasets
* Independent external validation
* Improved model generalisation
* Additional acoustic feature representations
* Alternative deep learning architectures
* Improved real-time processing
* Explainable AI techniques
* Continuous respiratory monitoring
* Further clinical validation

## Academic Context

This repository was developed as part of an **MSc Artificial Intelligence research project at Nottingham Trent University**.

The work combines machine learning, deep learning, audio signal processing, and application development to investigate the potential of respiratory acoustic analysis.


## Author

**Aditi Potphode**

MSc Artificial Intelligence
Nottingham Trent University

GitHub: [@potphodeaditi](https://github.com/potphodeaditi)

## Citation

If you use the code or research materials from this repository, please cite the project as:

```text
Potphode, A. (2026).
Deep Learning for Respiratory Sound Classification and Continuous Monitoring.
MSc Artificial Intelligence Research Project.
Nottingham Trent University.
```

## Disclaimer

This project has been developed for **academic and research purposes only**.

The predictions produced by the models should not be interpreted as medical advice or a clinical diagnosis and should not replace assessment by a qualified healthcare professional.

## Acknowledgements

I would like to acknowledge the researchers and organisations responsible for the respiratory sound datasets and open-source software libraries used in this project.

I also acknowledge the academic supervision and support provided during the MSc research project.


**© 2026 Aditi Potphode — MSc Artificial Intelligence Research Project**
