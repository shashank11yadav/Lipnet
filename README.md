# LipNet Implementation

## Table of Contents
- [Overview](#overview)
- [Technologies and Libraries Used](#technologies-and-libraries-used)
- [How to Run](#how-to-run)
- [Detailed Code Explanation](#detailed-code-explanation)
  - [Imports and Installs](#imports-and-installs)
  - [Dataset and Preprocessing](#dataset-and-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
  - [Final Model Saving](#final-model-saving)
- [Results](#results)
- [Why These Libraries?](#why-these-libraries)

## Overview
LipNet is a deep learning-based model for lipreading, inspired by the research paper [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599). This implementation uses a subset of the GRID corpus dataset for training and testing. The final model achieves an accuracy of **86%**.

## Technologies and Libraries Used
- Python
- TensorFlow/Keras
- OpenCV
- Matplotlib
- Scikit-learn
- ImageIO
- Gdown

## How to Run
1. Install the required dependencies:
   ```bash
   pip install matplotlib opencv-python mediapipe sklearn imageio gdown tensorflow
   ```
2. Download and extract the dataset:
   ```python
   import gdown, zipfile, os
   file_id = '1aS6SU0QhmaDrFaabcoXX8tY-DKNE7nTq'
   gdown.download(f'https://drive.google.com/uc?id={file_id}', 'data.zip', quiet=True)
   with zipfile.ZipFile('data.zip', 'r') as zip_ref:
       zip_ref.extractall('data')
   os.remove('data.zip')
   ```
3. Run the training script to train the model.
4. Use the saved model for evaluation on new data.

## Detailed Code Explanation

### Imports and Installs
The script starts with installing and importing necessary libraries, including TensorFlow, OpenCV, and Mediapipe for preprocessing and model development.

### Dataset and Preprocessing
- The model uses a subset of the [GRID corpus dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/).
- Video frames are extracted, converted to grayscale, and normalized.
- Mouth regions are extracted using OpenCV.

### Model Architecture
- A convolutional-recurrent neural network with layers:
  - **Conv3D + MaxPooling** (Feature extraction)
  - **Bidirectional LSTM** (Temporal sequence learning)
  - **Dense + Softmax** (Final classification)

### Training and Evaluation
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Evaluation Metrics**: Accuracy (86%)

### Final Model Saving
- The trained model is saved using TensorFlow’s `ModelCheckpoint`.

## Results
- The model achieves **86% accuracy** on the test dataset.
- Performance can be improved with more data and hyperparameter tuning.

## Why These Libraries?
- **TensorFlow/Keras**: Deep learning framework for model training.
- **OpenCV**: Video and image processing.
- **Scikit-learn**: Evaluation and preprocessing.
- **Gdown**: Downloading dataset from Google Drive.
- **Matplotlib**: Visualization of results.

