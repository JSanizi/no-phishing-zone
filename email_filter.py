import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from preprocess_mails import preprocess_email
from connect_to_email import get_unread_emails
from autoencoder import Autoencoder
from sending_mail import send_email


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
encoding_dim = 8  # Size of the latent space representation

# Re-initialize the Autoencoder with the correct input dimension
autoencoder = Autoencoder(input_dim, encoding_dim)

# Load the trained model
autoencoder = torch.load("models/autoencoder_model.pth", weights_only=False)

# Set the model to evaluation mode
autoencoder.eval()

# Spam detection function
def spam_detection_one_by_one(emails):
    spam_count = 0
    non_spam_count = 0
    predicted_label = [] 
    threshold = 0.0012

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
            predicted_label.append('Spam')
        else:
            print("✅ This email is likely NOT spam.")
            non_spam_count += 1
            predicted_label.append('Non-Spam')

    # Initialize the DataFrame to store predicted labels with two columns: email_number and label
    pred_email_df = pd.DataFrame({
        'email_number': range(1, len(email_texts) + 1),
        'label': predicted_label ,
    })

    # Save the predicted labels to a CSV file
    if os.path.exists('email_labels/predicted_sent_emails.csv'):
        os.remove('email_labels/predicted_sent_emails.csv')
    pred_email_df.to_csv('email_labels/predicted_sent_emails.csv', index=False)


    print(f"\n📊 Summary:")
    print(f"🚨 Spam emails: {spam_count}")
    print(f"✅ Non-spam emails: {non_spam_count}")


# Calculating the accuracy of the model
def calculate_accuracy():
    y_true_df = pd.read_csv('email_labels/true_sent_emails.csv')
    y_pred_df = pd.read_csv('email_labels/predicted_sent_emails.csv')

    y_true = y_true_df['true_label'].values
    y_pred = y_pred_df['label'].values

    print(f"\n📊 True labels: {y_true}")
    print(f"📊 Predicted labels: {y_pred}")

    accuracy = np.mean(y_true == y_pred) * 100
    print(f"\n📈 Accuracy: {accuracy:.2f}%")

    print(f"📊 True Positives: {np.sum((y_true == 'Spam') & (y_pred == 'Spam'))}")
    print(f"📊 True Negatives: {np.sum((y_true == 'Non-Spam') & (y_pred == 'Non-Spam'))}")
    print(f"📊 False Positives: {np.sum((y_true == 'Non-Spam') & (y_pred == 'Spam'))}")
    print(f"📊 False Negatives: {np.sum((y_true == 'Spam') & (y_pred == 'Non-Spam'))}")


# Run the spam detection
spam_detection_one_by_one(email_text_tensor)
calculate_accuracy()
