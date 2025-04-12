import torch
import torch.nn as nn
import numpy as np
from preprocess_mails import preprocess_email
from connect_to_email import get_unread_emails
from autoencoder import Autoencoder

# Fetch the unread emails
email_texts = get_unread_emails()

if not email_texts:
    print("📭 No email to process.")
    exit()

# Preprocess each email text
preprocessed_emails = []
for i, email_text in enumerate(email_texts):
    try:
        vector = preprocess_email(email_text)
        preprocessed_emails.append(vector)
    except Exception as e:
        print(f"🚨 Failed to preprocess email #{i+1}: {e}")


# Convert the list of numpy arrays into a 2D numpy array
preprocessed_emails = np.array(preprocessed_emails)

# Convert list of vectors into tensor
email_text_tensor = torch.tensor(preprocessed_emails, dtype=torch.float32)

# Define input dimension based on the vectorized email text
input_dim = email_text_tensor.shape[1]  # This should match the size of your TF-IDF features

# Re-initialize the Autoencoder with the correct input dimension
autoencoder = Autoencoder(input_dim)

# Load the trained model
autoencoder = torch.load("autoencoder_model.pth", weights_only=False)

# Set the model to evaluation mode
autoencoder.eval()

# Spam detection function
def spam_detection_one_by_one(emails):
    spam_count = 0
    non_spam_count = 0
    threshold = 0.048

    for i, email in enumerate(emails):
        email = email.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            reconstructed = autoencoder(email)

        reconstructed = reconstructed.squeeze(0)
        reconstruction_error = torch.mean((email.squeeze(0) - reconstructed) ** 2).item()

        print(f"\n📧 Email #{i+1} reconstruction error: {reconstruction_error:.6f}")

        if reconstruction_error > threshold:
            print("⚠️ This email is likely SPAM.")
            spam_count += 1
        else:
            print("✅ This email is likely NOT spam.")
            non_spam_count += 1

    print(f"\n📊 Summary:")
    print(f"🚨 Spam emails: {spam_count}")
    print(f"✅ Non-spam emails: {non_spam_count}")

# Run the spam detection
spam_detection_one_by_one(email_text_tensor)
