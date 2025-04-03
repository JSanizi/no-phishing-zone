import torch
import torch.nn as nn
import torch.optim as optim
from preprocess_mails import preprocess_email
from connect_to_email import get_latest_email

# Fetch the latest email
email_text = get_latest_email()

if email_text is None:
    print("No email to process.")
    exit()

# print email text
print("Raw email text:", email_text)

# Preprocess the email text
email_text_vectorized = preprocess_email(email_text)

print("Preprocessed email text:", email_text_vectorized)

# Convert the data to PyTorch tensors
email_text_tensor = torch.tensor(email_text_vectorized, dtype=torch.float32)


# Load the model from the previous training
autoencoder = torch.load("autoencoder_model.pth", weights_only=False)

# Set the model to evaluation mode
autoencoder.eval()

# Define a function to filter spam emails
def spam_detection(email_text_tensor, threshold=0.1):
    with torch.no_grad():
        reconstructed = autoencoder(email_text_tensor)
    
    # Compute the reconstruction error
    reconstruction_error = torch.mean((email_text_tensor - reconstructed) ** 2).item()
    print("Reconstruction error:", reconstruction_error)

    return reconstruction_error > threshold  # Define a threshold for spam detection

# Example usage
if spam_detection is True:
    print("🚨 This email is likely spam!")
else:
    print("✅ This email is likely NOT spam.")