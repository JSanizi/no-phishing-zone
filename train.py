import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

from autoencoder import Autoencoder
from split_data import split_dataset
from data_cleaning import clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


# Split dataset into train, validation, and test sets
train_df, val_df, test_df = split_dataset('datasets/SpamAssasin.csv')

# Clean the data
train, train_labels = clean_data(train_df)
val, val_labels = clean_data(val_df)
test, test_labels = clean_data(test_df)


# Vectorize the data using TF-IDF
vectorizer = TfidfVectorizer()  # Adjust max_features as needed
train_vectorized = vectorizer.fit_transform(train).toarray()
val_vectorized = vectorizer.transform(val).toarray()
test_vectorized = vectorizer.transform(test).toarray()


# Convert the data arrays to PyTorch tensors
train_tensor = torch.tensor(train_vectorized, dtype=torch.float32)
val_tensor = torch.tensor(val_vectorized, dtype=torch.float32)
test_tensor = torch.tensor(test_vectorized, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

# Save the tensor for testing
torch.save(train_tensor, 'train_tensor.pth')
torch.save(val_tensor, 'val_tensor.pth')
torch.save(test_tensor, 'test_tensor.pth')
torch.save(val_labels_tensor, 'val_labels_tensor.pth')
torch.save(test_labels_tensor, 'test_labels_tensor.pth')

# Model initialization
input_dim = train_vectorized.shape[1] # This should match the size of your TF-IDF features


# Hyperparameters
learning_rate = 0.01  # Learning rate can be adjusted
epochs = 10 # Number of epochs for training
batch_size = 32  # Batch size for trainin
encoding_dim = 8 # Dimension of the encoding layer
weight_decay = 0.0001 # Weight decay for regularization

# Model, Loss, Optimizer
model = Autoencoder(input_dim, encoding_dim)  # Initialize the model
criterion = nn.MSELoss() # Mean Squared Error Loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam optimizer with weight decay

# Dataloader
train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_tensor, val_labels_tensor)  # Create a dataset with features and labels
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_tensor, test_labels_tensor)  # Create a dataset with features and labels
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Lists to store losses for plotting
train_loss_per_epoch = []
val_loss_per_epoch = []
test_loss_per_epoch = []
true_labels = []  # To store true labels for evaluation
test_losses = []  # To store test losses for evaluation


# Training Loop
for epoch in range(epochs):
    # Initiate losses
    train_losses = []  # To store the loss values for plotting
    val_losses = [] # To store the loss values for plotting
    all_losses = []  # To store the loss values for each batch
    total_loss = 0  # Initialize total loss for the epoch

    model.train()  # Set the model to training mode
    for batch in train_loader:
        inputs = batch[0] # Get the batch data

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Calculate the loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        total_loss += loss.item()  # Accumulate the loss

    # Average loss
    avg_loss = total_loss / len(train_loader)
    train_loss_per_epoch.append(avg_loss)  # Store the average loss for plotting

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)  # Forward pass
            loss_per_sample = torch.mean((outputs - inputs) ** 2, dim=1)
            # Only non-spam (label == 0)
            val_losses.extend(loss_per_sample[labels == 0].cpu().numpy())

    # Set threshold based on mean + 2*std of non-spam
    threshold = 0.01
    print(f"\n🔥 Detection threshold (based on non-spam): {threshold:.6f}")

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss_per_sample = torch.mean((outputs - inputs) ** 2, dim=1)
            test_losses.extend(loss_per_sample.cpu().numpy()) # Store the test losses
            true_labels.extend(labels.cpu().numpy()) # Store the true labels

    # Dynamically calculate the threshold based on validation losses
    threshold = np.mean(val_losses) + 2 * np.std(val_losses)
    print(f"\n🔥 Updated Detection Threshold: {threshold:.6f}")

# Predict based on threshold
predicted_labels = [1 if loss > threshold else 0 for loss in test_losses]

# Suppress warnings in classification report
print("\n📊 Classification Report (Spam Detection using Autoencoder):")
print(classification_report(true_labels, predicted_labels, target_names=["Non-Spam", "Spam"], zero_division=0))

# Plot reconstruction loss distributions for Spam and Non-Spam
non_spam_losses = [loss for loss, label in zip(test_losses, true_labels) if label == 0]
spam_losses = [loss for loss, label in zip(test_losses, true_labels) if label == 1]

plt.figure()
plt.hist(non_spam_losses, bins=50, alpha=0.7, label='Non-Spam')
plt.hist(spam_losses, bins=50, alpha=0.7, label='Spam')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Loss")
plt.ylabel("Frequency")
plt.title("Reconstruction Loss Distribution")
plt.legend()
plt.show()

# Calculate and print True Negative Rate (TNR) and False Positive Rate (FPR)
cm = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = cm.ravel()

tnr = tn / (tn + fp)  # True Negative Rate
fpr = fp / (tn + fp)  # False Positive Rate

print(f"True Negative Rate (TNR): {tnr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")

# Save confusion matrix plot
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(f"Graphs/confusion_matrix_epoch_{epoch+1}.png")
plt.show()