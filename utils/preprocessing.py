from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fit_transform(self, X):
        # First convert categorical variables to numeric using Label Encoding
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
        X = X.copy()
        
        # Label encode categorical columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = pd.to_numeric(X[col], errors='coerce')  # Convert to numeric, set errors as NaN
            X[col] = X[col].fillna(X[col].median())  # Fill NaN with median
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Scale numerical features
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        return X.astype(np.float32).values

    def transform(self, X):
        # Transform new data using fitted encoders and scaler
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
        X = X.copy()
        
        # Label encode categorical columns using fitted encoders
        for col in categorical_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())
            X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features using fitted scaler
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X.astype(np.float32).values