import os
import pickle
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from training_and_tunning_models.data_preprocessing import clean_data
from training_and_tunning_models.autoencoder import Autoencoder

# Load the best threshold for the autoencoder
with open("models/autoencoder_param_tuning_results.pkl", "rb") as f:
    ac_tuning_results = pickle.load(f)

autoencoder_threshold = ac_tuning_results.get("best_threshold", None)  # fallback if not found
best_ac_params = ac_tuning_results.get("best_params", None)

def find_threshold_non_spam(autoencoder, email_text_tensor):
    autoencoder.eval()

    reconstruction_errors = []
    for i, email in enumerate(email_text_tensor):
        with torch.no_grad():
            reconstructed = autoencoder(email_text_tensor)

        # Compute the reconstruction error
        reconstruction_error = torch.mean((email - reconstructed) ** 2).item()
        reconstruction_errors.append(reconstruction_error)
    
    # Load true labels
    y_true_df = pd.read_csv('email_labels/true_sent_emails.csv')
    y_true = y_true_df['true_label'].values

    # find the lowest reconstruction error for non-spam emails using the true label to match the email number with the reconstruction error for i.
    non_spam_errors = [reconstruction_errors[i] for i in range(len(y_true)) if y_true[i] == 'Non-Spam']
    non_spam_errors.sort()
    threshold = round(non_spam_errors[int(len(non_spam_errors) * 0.95)], 6)  # 95th percentile of non-spam errors, rounded to 6 digits
    # print(f"🔍 Using threshold: {threshold:.6f}")
    
    return threshold
    

def run_autoencoder(preprocessed_emails):
    # Clean data
    clean_email = clean_data(preprocessed_emails)

    # Vectorize the email text
    vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    preprocessed_emails = vectorizer.transform(clean_email)

    # Convert list of vectors into tensor
    email_text_tensor = torch.tensor(preprocessed_emails.toarray(), dtype=torch.float32)

    # Define input dimension based on the vectorized email text
    input_dim = email_text_tensor.shape[1]  # This should match the size of your TF-IDF features
    encoding_dim = best_ac_params["encoding_dim"]  # Size of the latent space representation

    # Re-initialize the Autoencoder with the correct input dimension
    autoencoder = Autoencoder(input_dim, encoding_dim)

    # Load the trained model
    autoencoder.load_state_dict(torch.load("models/autoencoder_best_model.pt"))
    
    # Set the model to evaluation mode
    autoencoder.eval()

    # Initiate reconstruction error and threshold as global variables
    reconstruction_errors = []
    threshold = find_threshold_non_spam(autoencoder, email_text_tensor) # Use the threshold from the training phase
    """ print(f"\n🔍 Using threshold: {threshold:.6f}\n"
          f"📊 Encoding dimension: {encoding_dim}\n"
          f"📊 Input dimension: {input_dim}\n"
          f"📊 Number of emails: {len(email_text_tensor)}\n")"""
    
    spam_count = 0
    non_spam_count = 0
    predicted_label = []

    for i, email in enumerate(email_text_tensor):
        with torch.no_grad():
            reconstructed = autoencoder(email_text_tensor)

        # Compute the reconstruction error
        reconstruction_error = torch.mean((email - reconstructed) ** 2).item()
        reconstruction_errors.append(reconstruction_error)

        # print(f"\n📧 Email #{i+1} reconstruction error: {reconstruction_error:.6f}")

        if reconstruction_error > threshold:
            # print("⚠️ This email is likely SPAM.")
            spam_count += 1
            predicted_label.append('Spam')
        else:
            # print("✅ This email is likely NOT spam.")
            non_spam_count += 1
            predicted_label.append('Non-Spam')

    # Initialize the DataFrame to store predicted labels with two columns: email_number and label
    pred_email_df = pd.DataFrame({
        'email_number': range(1, len(predicted_label) + 1),
        'label': predicted_label,
    })

    # Save the predicted labels to a CSV file
    if os.path.exists('email_labels/Autoencoder_predicted_sent_emails.csv'):
        os.remove('email_labels/Autoencoder_predicted_sent_emails.csv')
    pred_email_df.to_csv('email_labels/Autoencoder_predicted_sent_emails.csv', index=False)


    # print(f"\n📊 Summary:")
    # print(f"🚨 Spam emails: {spam_count}")
    # print(f"✅ Non-spam emails: {non_spam_count}")

    calculate_accuracy(reconstruction_errors, threshold)



# Calculating the accuracy of the model
def calculate_accuracy(reconstruction_errors, threshold):
    y_true_df = pd.read_csv('email_labels/true_sent_emails.csv')
    y_pred_df = pd.read_csv('email_labels/Autoencoder_predicted_sent_emails.csv')

    y_true = y_true_df['true_label'].values
    y_pred = y_pred_df['label'].values

    """# show confusion matrix.
    print("\n📊 Confusion Matrix for Autoencoder:")
    print(confusion_matrix(y_true, y_pred))
    print("\n📈 Classification Report for Autoencoder:")
    print(classification_report(y_true, y_pred, target_names=['Non-Spam', 'Spam'], zero_division=0))"""

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()  # Extract values from the confusion matrix
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
    plt.title(f'Confusion Matrix with {len(y_true)} emails\n')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if os.path.exists(f'graphs/confusion/{len(y_true)}_emails/autoencoder_confusion_matrix.png'):
        os.remove(f'graphs/confusion/{len(y_true)}_emails/autoencoder_confusion_matrix.png')
    plt.savefig(f'graphs/confusion/{len(y_true)}_emails/autoencoder_confusion_matrix.png')
    plt.close()

    # Save the classification report as a text file
    if os.path.exists(f'graphs/classification_reports/{len(y_true)}_emails/autoencoder_classification_report.txt'):
        os.remove(f'graphs/classification_reports/{len(y_true)}_emails/autoencoder_classification_report.txt')

    report = classification_report(y_true, y_pred, target_names=['Non-Spam', 'Spam'], zero_division=0)
    with open(f'graphs/classification_reports/{len(y_true)}_emails/autoencoder_classification_report.txt', 'w') as f:
        f.write(f'Amount of emails: {len(y_true)}\n')
        f.write(report)
    print(f"📄 Classification report saved as autoencoder_classification_report.txt")