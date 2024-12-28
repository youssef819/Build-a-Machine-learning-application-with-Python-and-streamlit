# Mushroom Classification App 🍄

This project demonstrates a machine learning application implemented using Streamlit. It serves as a practical example to master the functionalities of this library while benchmarking some popular machine learning algorithms.

## Overview

The application classifies mushrooms as either edible or poisonous using three different machine learning models:
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest Classifier

Users can select the classifier, adjust hyperparameters, and choose evaluation metrics to visualize the model's performance.

## Features

- Interactive model selection
- Real-time hyperparameter tuning
- Performance visualization and metrics
- Cross-validation results
- Model comparison capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mushroom-classifier.git
cd mushroom-classifier
```
2. Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
3. Install the required packages:
```bash   
pip install -r requirements.txt
```
## Usage
1. Run the application:
```bash
streamlit run app.py
```
2. Open your web browser and navigate to `http://localhost:8501`
  
## Model Performance

Each classifier has been trained and evaluated on the mushroom dataset with the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
