import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
import os

# Make sure the models directory exists
os.makedirs('models', exist_ok=True)

print("⏳ Fetching live dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

print("🛠️ Preprocessing and Feature Engineering...")
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

df['Age_BMI_Index'] = df['Age'] * df['BMI']

X = df.drop(columns=['Outcome', 'SkinThickness'])
y = df['Outcome']

X = X[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Age_BMI_Index']]

print("🚀 Training production-grade XGBoost Model...")
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss')
model.fit(X, y)

joblib.dump(model, 'models/best_xgboost_model.pkl')
print("✅ Model successfully trained and saved to 'models/best_xgboost_model.pkl'!")