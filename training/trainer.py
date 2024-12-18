import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_dataset, batch_size, learning_rate, num_epochs, val_split=0.2):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Split dataset into train and validation
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_data, self.val_data = random_split(train_dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        
        # For storing metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Store hyperparameters
        self.hyperparameters = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'val_split': val_split,
            'device': str(self.device)
        }
        
    def train(self):
        logger.info(f"Training on device: {self.device}")
        
        training_history = {
            'epochs': [],
            'final_metrics': {}
        }
        
        for epoch in range(self.num_epochs):
            epoch_metrics = {'epoch': epoch + 1}
            
            # Training phase
            self.model.train()
            train_loss, train_acc = self._train_epoch()
            
            # Validation phase
            self.model.eval()
            val_loss, val_acc = self._validate()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Store epoch metrics
            epoch_metrics.update({
                'train_loss': float(train_loss),
                'train_accuracy': float(train_acc),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_acc)
            })
            training_history['epochs'].append(epoch_metrics)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # After training, plot results and save metrics
        self.plot_training_history()
        final_metrics = self.evaluate_model()
        training_history['final_metrics'] = final_metrics
        self.save_results(training_history)
    
    def _train_epoch(self):
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = accuracy_score(true_labels, predictions)
        return epoch_loss, epoch_accuracy
    
    def _validate(self):
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                true_labels.extend(batch_y.cpu().numpy())
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_accuracy = accuracy_score(true_labels, predictions)
        return epoch_loss, epoch_accuracy
    
    def plot_training_history(self):
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def evaluate_model(self):
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Get classification report as dict
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return {
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
    
    def save_results(self, training_history):
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'hyperparameters': self.hyperparameters,
            'training_history': training_history,
            'model_summary': str(self.model),
            'final_metrics': {
                'train_loss': float(self.train_losses[-1]),
                'train_accuracy': float(self.train_accuracies[-1]),
                'val_loss': float(self.val_losses[-1]),
                'val_accuracy': float(self.val_accuracies[-1])
            }
        }
        
        # Save to JSON file
        filename = f'training_results_{results["timestamp"]}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {filename}")