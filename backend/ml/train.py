import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic health data for training"""
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(45, 15, n_samples)
    gender = np.random.binomial(1, 0.5, n_samples)
    height = np.random.normal(170, 10, n_samples)
    weight = np.random.normal(70, 15, n_samples)
    exercise = np.random.random(n_samples)
    smoking = np.random.random(n_samples)
    alcohol = np.random.random(n_samples)
    
    # Generate medical conditions (binary features)
    conditions = np.random.binomial(1, 0.2, (n_samples, 4))
    family_history = np.random.binomial(1, 0.3, (n_samples, 4))
    
    # Combine all features
    X = np.column_stack([
        age/100, gender, height/200, weight/150,
        exercise, smoking, alcohol,
        conditions, family_history
    ])
    
    # Generate risk scores based on feature combinations
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Calculate base risk from age and lifestyle factors
        risk = (
            0.3 * (age[i]/100) +
            0.2 * smoking[i] +
            0.2 * alcohol[i] +
            0.1 * (1 - exercise[i]) +
            0.2 * np.mean(conditions[i]) +
            0.2 * np.mean(family_history[i])
        )
        
        # Add BMI factor
        bmi = weight[i] / ((height[i]/100) ** 2)
        if bmi < 18.5 or bmi > 30:
            risk += 0.2
        
        y[i] = min(1.0, risk)
    
    return X, y

def train_model():
    """Train the health risk prediction model"""
    # Generate synthetic training data
    X, y = generate_synthetic_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train random forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Create models directory if it doesn't exist
    os.makedirs("../models", exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, "../models/health_risk_model.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")

if __name__ == "__main__":
    train_model()
    print("Model training completed successfully!")