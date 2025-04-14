import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import pandas as pd

from autoencoder import Autoencoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


# Define the grid of hyperparameters
param_grid = {
    'encoding_dim': [8, 16, 32],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'weight_decay': [0, 1e-5, 1e-4]
}

# Function to train and evaluate the model
def train_and_evaluate(train_tensor, val_tensor, val_labels_tensor, input_dim, params):
    # Unpack hyperparameters
    encoding_dim = params['encoding_dim']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    weight_decay = params['weight_decay']
    
    # Initialize the model, loss, and optimizer
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss_per_sample = torch.mean((outputs - inputs) ** 2, dim=1)
            val_losses.extend(loss_per_sample[labels == 0].cpu().numpy())
    
    # Return the average validation loss
    avg_val_loss = np.mean(val_losses)
    return avg_val_loss

# Perform grid search
def grid_search(train_tensor, val_tensor, val_labels_tensor, input_dim, param_grid):
    best_params = None
    best_val_loss = float('inf')
    
    # Iterate over all combinations of hyperparameters
    for encoding_dim in param_grid['encoding_dim']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for weight_decay in param_grid['weight_decay']:
                    params = {
                        'encoding_dim': encoding_dim,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'weight_decay': weight_decay
                    }
                    print(f"Testing parameters: {params}")
                    avg_val_loss = train_and_evaluate(train_tensor, val_tensor, val_labels_tensor, input_dim, params)
                    print(f"Validation Loss: {avg_val_loss:.4f}")
                    
                    # Update the best parameters if the current configuration is better
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_params = params
    
    print("\nBest Parameters:")
    print(best_params)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return best_params

# Save the classification report to a text file
true_labels = np.array([])  # Replace with actual true labels
predicted_labels = np.array([])  # Replace with actual predicted labels

with open("classification_report.txt", "w") as f:
    report = classification_report(true_labels, predicted_labels, target_names=["Non-Spam", "Spam"])
    f.write("📊 Classification Report (Spam Detection using Autoencoder):\n")
    f.write(report)
print("Classification report saved to 'classification_report.txt'")

# Save the confusion matrix to a CSV file
cm = confusion_matrix(true_labels, predicted_labels)  # Ensure true_labels and predicted_labels are defined
cm_df = pd.DataFrame(cm, index=['Non-Spam', 'Spam'], columns=['Non-Spam', 'Spam'])
cm_df.to_csv("confusion_matrix.csv")
print("Confusion matrix saved to 'confusion_matrix.csv'")

# Save the best hyperparameters to a text file
best_epoch = 1  # Replace with actual best epoch
train_loss_per_epoch = []  # Replace with actual training loss per epoch
val_loss_per_epoch = []  # Replace with actual validation loss per epoch
test_loss_per_epoch = []  # Replace with actual test loss per epoch
threshold = 0.5  # Replace with actual threshold
encoding_dim = 16  # Replace with actual best encoding dimension
learning_rate = 0.001  # Replace with actual best learning rate
batch_size = 32  # Replace with actual best batch size
weight_decay = 1e-5  # Replace with actual best weight decay

with open("best_hyperparameters.txt", "w") as f:
    f.write("Best Hyperparameters:\n")
    f.write(f"Encoding Dimension: {encoding_dim}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Weight Decay: {weight_decay}\n")
print("Best hyperparameters saved to 'best_hyperparameters.txt'")

# Save results to a JSON file
results_json = {
    "Best Epoch": best_epoch,
    "Best Test Loss": test_loss_per_epoch[best_epoch - 1] if test_loss_per_epoch else None,
    "Threshold": threshold,
    "Train Loss Per Epoch": train_loss_per_epoch,
    "Validation Loss Per Epoch": val_loss_per_epoch,
    "Test Loss Per Epoch": test_loss_per_epoch
}

with open("results.json", "w") as f:
    json.dump(results_json, f, indent=4)
print("Results saved to 'results.json'")

# Save the results as a CSV file
epochs = len(train_loss_per_epoch)  # Ensure epochs is defined
results = {
    'Epoch': np.arange(1, epochs + 1),
    'Train Loss': train_loss_per_epoch,
    'Validation Loss': val_loss_per_epoch,
    'Test Loss': test_loss_per_epoch
}
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)
print("Results saved to 'results.csv'")

# Example usage
if __name__ == "__main__":
    # Load your preprocessed tensors (replace with your actual data loading code)
    train_tensor = torch.load("train_tensor.pth")
    val_tensor = torch.load("val_tensor.pth")
    val_labels_tensor = torch.load("val_labels_tensor.pth")
    input_dim = train_tensor.shape[1]
    
    # Perform grid search
    best_params = grid_search(train_tensor, val_tensor, val_labels_tensor, input_dim, param_grid)

