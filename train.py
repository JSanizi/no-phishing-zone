import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from autoencoder import Autoencoder
from dataprocess.data_preprocess import load_data
from sklearn.model_selection import train_test_split

# Load data from preprocess.py
X_non_spam_vectorized, X_spam_vectorized = load_data()

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Learning rate can be adjusted

# Track losses
train_losses = []
val_losses = []

# Training Loop
epochs = 30  # Number of epochs can be adjusted
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    train_output = model(X_train_tensor)
    train_loss = criterion(train_output, X_train_tensor)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = criterion(val_output, X_val_tensor)
        
    #Store losses
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    # Print loss every 5 epochs
    if (epoch + 1) % 5 == 0:  # Print loss every 5 epochs
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Final Test Evaluation
model.eval()
with torch.no_grad():
    test_output = model(X_test_tensor)
    test_loss = criterion(test_output, X_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Save the model
torch.save(model, 'autoencoder_model.pth')
print("Model saved as 'autoencoder_model.pth'")
"""---------------------------------------------------------------------------------------------------------------"""
# Visualizations

# 1️⃣ Plot Loss Curve 📉
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), val_losses, label='Validation Loss', color='red', linestyle='-')
plt.plot(range(epochs), train_losses, label='Train Loss', color='blue', linestyle='--')
plt.ylabel('Epochs')
plt.xlabel('Loss')
plt.title(f'Training vs. Validation Loss' + "\n" + f"Test Loss: {test_loss.item():.4f}")
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



""""
# 2️⃣ Latent Space Visualization 🌌 (PCA & t-SNE)
encoded_train = model.encoder(X_train_tensor).detach().numpy()

# PCA for 2D Projection
pca = PCA(n_components=2)
pca_result = pca.fit_transform(encoded_train)

plt.figure(figsize=(8, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, color='blue')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Latent Space Representation (PCA)")
plt.show()

# t-SNE for another perspective
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(encoded_train)

plt.figure(figsize=(8, 5))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, color='green')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Latent Space Representation (t-SNE)")
plt.show()

# 3️⃣ Compare Original vs. Reconstructed Features 🎭
num_samples = 5
sample_indices = np.random.choice(len(X_test_tensor), num_samples, replace=False)

print("\n📌 **Original vs. Reconstructed Text Features**")
for i in sample_indices:
    print(f"\n🔹 **Sample {i}**")
    print(f"Original: {X_test_tensor[i].numpy()[:10]} ...")
    print(f"Reconstructed: {test_output[i].numpy()[:10]} ...")"""