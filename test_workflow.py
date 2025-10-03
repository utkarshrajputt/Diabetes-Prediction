# Test Script for Diabetes Prediction Project
# This script tests the workflow without running the full notebook

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import pickle

print("🧪 Testing Diabetes Prediction Workflow...")

# Load and prepare data
print("\n1️⃣ Loading dataset...")
try:
    df = pd.read_csv('diabetes.csv')
    print(f"✅ Dataset loaded: {df.shape}")
except FileNotFoundError:
    print("❌ diabetes.csv not found!")
    exit(1)

# Basic preprocessing
print("\n2️⃣ Preprocessing data...")
df_processed = df.copy()

# Handle zero values in certain columns
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    if col in df_processed.columns:
        median_value = df_processed[df_processed[col] != 0][col].median()
        df_processed[col] = df_processed[col].replace(0, median_value)

# Prepare features and target
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df_processed[features]
y = df_processed['Outcome']

print("✅ Data preprocessed successfully")

# Split data
print("\n3️⃣ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training set: {X_train.shape[0]}, Testing set: {X_test.shape[0]}")

# Scale features
print("\n4️⃣ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# Train a simple model
print("\n5️⃣ Training model...")
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Model trained - Accuracy: {accuracy:.3f}, F1-Score: {f1:.3f}")

# Save model and scaler
print("\n6️⃣ Saving model and scaler...")
try:
    joblib.dump(model, 'best_diabetes_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("✅ Model and scaler saved successfully")
except Exception as e:
    print(f"❌ Error saving files: {e}")
    exit(1)

# Test loading
print("\n7️⃣ Testing model loading...")
try:
    loaded_model = joblib.load('best_diabetes_model.pkl')
    loaded_scaler = joblib.load('feature_scaler.pkl')
    
    with open('feature_names.pkl', 'rb') as f:
        loaded_features = pickle.load(f)
    
    print("✅ Model and scaler loaded successfully")
    print(f"✅ Feature names: {loaded_features}")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit(1)

# Test prediction
print("\n8️⃣ Testing prediction...")
try:
    # Create sample input
    sample_input = pd.DataFrame({
        'Pregnancies': [1],
        'Glucose': [120],
        'BloodPressure': [80],
        'SkinThickness': [25],
        'Insulin': [100],
        'BMI': [25.0],
        'DiabetesPedigreeFunction': [0.5],
        'Age': [30]
    })
    
    # Scale and predict
    sample_scaled = loaded_scaler.transform(sample_input)
    prediction = loaded_model.predict(sample_scaled)[0]
    probability = loaded_model.predict_proba(sample_scaled)[0][1]
    
    print(f"✅ Sample prediction: {prediction}")
    print(f"✅ Sample probability: {probability:.3f}")
    
except Exception as e:
    print(f"❌ Error making prediction: {e}")
    exit(1)

print("\n🎉 All tests passed! The workflow is ready.")
print("\n📋 Next steps:")
print("1. Run the Jupyter notebook for full analysis")
print("2. Launch Streamlit app with: streamlit run app.py")
print("3. Open the app in your browser for interactive predictions")