from data.data_loader import DataLoader
from models.dataset import HeartDiseaseDataset
from models.transformer_model import TabularTransformer
from training.trainer import Trainer
from utils.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split


def main():
    # Initialize data loader
    data_loader = DataLoader()
    X, y = data_loader.load_data()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess data
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)  # Use transform instead of fit_transform for test data
    
    # Create datasets
    train_dataset = HeartDiseaseDataset(X_train_processed, y_train)
    test_dataset = HeartDiseaseDataset(X_test_processed, y_test)
    
    # Initialize model
    model = TabularTransformer(
        input_dim=X_train_processed.shape[1],
        num_classes=2,
        dim=32,
        depth=3,
        heads=4,
        mlp_dim=64
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=50
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()


