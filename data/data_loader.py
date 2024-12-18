import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
    def load_data(self):
        # Load the Cleveland dataset
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                          names=self.columns + ['target'])
        
        # Clean the data
        data = data.replace('?', pd.NA)
        data = data.dropna()
        
        # Convert target to binary (0: no disease, 1: disease)
        data['target'] = (data['target'] > 0).astype(int)
        
        X = data[self.columns]
        y = data['target']
        
        return X, y 