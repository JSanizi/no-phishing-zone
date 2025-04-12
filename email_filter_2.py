import torch
import torch.nn as nn
import torch.optim as optim

from dataprocess.data_preprocess_ling import load_data
from autoencoder import Autoencoder

# Preprocess the email text
X_non_phishing_vectorized, X_phishing_vectorized, X_all_emails_vectorized = load_data()

# Convert the data to PyTorch tensors
X_all_emails_tensor = torch.tensor(X_all_emails_vectorized, dtype=torch.float32)

# Define input dimension based on the vectorized email text
input_dim = X_all_emails_tensor.shape[1]  # This should match the size of your TF-IDF features

# Re-initialize the Autoencoder with the correct input dimension
autoencoder = Autoencoder(input_dim)

# Load the model from the previous training
autoencoder = torch.load("autoencoder_model.pth", weights_only=False)

# Set the model to evaluation model
autoencoder.eval()

# Define a function to filter spam emails one by one and count the number of spam emails and non spam emails and print the number of spam emails and non spam emails
def spam_detection_one_by_one(emails):
    spam_count = 0
    non_spam_count = 0
    threshold = 0.048

    for email in emails:
        with torch.no_grad():
            reconstructed = autoencoder(email)

        # Compute the reconstruction error
        reconstruction_error = torch.mean((email - reconstructed) ** 2).item()

        if reconstruction_error > threshold:  # Define a threshold for spam detection
            spam_count += 1
        else:
            non_spam_count += 1

    print(f"🚨 Number of spam emails after training (with threshold {threshold}): {spam_count}")
    print(f"✅ Number of non-spam emails after training (with threshold {threshold}): {non_spam_count}")


# Example usage
spam_detection_one_by_one(X_all_emails_tensor)




"""def spam_detection(emails, threshold=0.029):
    with torch.no_grad():
        reconstructed = autoencoder(emails)

    # Compute the reconstruction error
    reconstruction_error = torch.mean((emails - reconstructed) ** 2).item()
    print("Reconstruction error:", reconstruction_error)

    return reconstruction_error > threshold  # Define a threshold for spam detection

# Example usage
if spam_detection(X_all_emails_tensor) is True:
    print("🚨 This email is likely spam!")
else:
    print("✅ This email is likely NOT spam.")"""