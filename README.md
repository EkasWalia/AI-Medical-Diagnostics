# 🩺 AI Medical Diagnostics - Diabetes Prediction using XGBoost

An end-to-end Machine Learning project that predicts the probability of diabetes using patient health parameters.

The project includes:

- Data preprocessing
- Feature engineering
- XGBoost model training
- Model serialization using Joblib
- Interactive Streamlit web application for real-time predictions


## Features

- Real-time diabetes prediction
- XGBoost Machine Learning model
- Automatic preprocessing of missing values
- Feature engineering for improved accuracy
- Probability-based predictions
- Clean Streamlit dashboard


## Tech Stack

- Python
- Pandas
- NumPy
- XGBoost
- Joblib
- Streamlit


## Dataset

Pima Indians Diabetes Dataset

Features include:

- Glucose
- Blood Pressure
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Engineered Feature:

- Age × BMI Index

Target:

- Diabetes (Yes / No)


## Machine Learning Pipeline

1. Load dataset
2. Replace invalid zero values
3. Fill missing values using median
4. Feature Engineering
5. Train XGBoost Classifier
6. Save trained model
7. Deploy using Streamlit


## Installation

Clone the repository

```bash
git clone https://github.com/EkasWalia/AI-Medical-Diagnostics.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Run the application

```bash
streamlit run app.py
```


## Future Improvements

- Hyperparameter tuning
- Cross Validation
- SHAP Explainability
- ROC Curve Analysis
- Model Comparison (Random Forest, SVM, Logistic Regression)
- Cloud Deployment


## Author

Made with ❤️ by Ekas Walia
