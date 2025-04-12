import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from dataprocess.data_preprocess_ling import load_data
from sklearn.model_selection import train_test_split

# Load data from preprocess.py
X_non_spam_vectorized, X_spam_vectorized, X_all_emails_vectorized = load_data()

# Split the data into training and testing sets
X_train, X_temp = train_test_split(X_non_spam_vectorized, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42, shuffle=True)

# Convert the data arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Model initialization
input_dim = X_train.shape[1]  # This should match the size of your TF-IDF features
model = Autoencoder(input_dim)

# Loss function and optimizer
learning_rate = 0.001  # Learning rate can be adjusted
criterion = nn.MSELoss() # Mean Squared Error Loss for reconstruction
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer with weight decay

# Track losses
train_losses = []
val_losses = []
epochs = 30

# Training Loop
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    train_output = model(X_train_tensor)
    train_loss = criterion(train_output, X_train_tensor)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    train_losses.append(train_loss.item())

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = criterion(val_output, X_val_tensor)
        
    #Store losses
    val_losses.append(val_loss.item())

    # Print loss every 1 epochs
    if (epoch + 1) % 1 == 0:  # Print loss every 1 epochs
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Final Test Evaluation
model.eval()
with torch.no_grad():
    test_output = model(X_test_tensor)
    test_loss = criterion(test_output, X_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Validation Loss: {val_loss.item():.4f}')
    print(f'Accuracy: {1 - val_loss.item():.4f}')

# Save the model
torch.save(model, 'autoencoder_model.pth')
print("Model saved as 'autoencoder_model.pth'")
"""-----------------------------------------------------------------------------------------------"""
# Visualizations

# 1️⃣ Plot Loss function 📉
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Training and Validation Loss over Epochs" + "\n" + f"Final Test Loss: {test_loss.item():.4f}")
plt.legend()

# Save the loss plot in the folder 'Graphs'
import os
if not os.path.exists('Graphs'):
    os.makedirs('Graphs')
plt.savefig(f"Graphs/loss_curve_epoch_{epoch+1}_test.png")

# If the file name exists, overwrite it
# Check if the file exists and overwrite it
if os.path.exists(f"Graphs/loss_curve_epoch_{epoch+1}_test.png"):
    os.remove(f"Graphs/loss_curve_epoch_{epoch+1}_test.png") # Remove the old file if it exists
plt.savefig(f"Graphs/loss_curve_epoch_{epoch+1}_test.png")   # Save the new file 
plt.show()

# 2️⃣ Plot learning rate over loss function 📈
plt.figure(figsize=(12, 6))
plt.plot(train_losses, [learning_rate] * len(train_losses), label='Train Loss', color='blue', marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Train Loss")
plt.title(f"Learning Rate vs Train Loss" + "\n" + f"Final Test Loss: {test_loss.item():.4f}")
plt.legend()

# Check if the file exists and overwrite it
if os.path.exists("Graphs/learning_rate.png"):
    os.remove("Graphs/learning_rate.png") # Remove the old file if it exists
plt.savefig("Graphs/learning_rate.png")   # Save the new file
plt.show()
