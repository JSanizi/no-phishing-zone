import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pickle

from autoencoder import Autoencoder
from split_data import split_dataset
from data_cleaning import clean_data, get_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Split dataset into train, validation, and test sets
# train_df, val_df, test_df, spam_emails_df = split_dataset('datasets/SpamAssasin.csv')
train_df, val_df, test_df, spam_emails_df = split_dataset('datasets/CEAS_08.csv')

# Clean the data
train, train_labels = clean_data(train_df)
test, test_labels = clean_data(test_df)

# Split train data into train and validation sets
x_non_spam_train_data, x_non_spam_val_data = train_test_split(train, test_size=0.25, random_state=42)

# Convert the data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
x_non_spam_train_data = vectorizer.fit_transform(x_non_spam_train_data).toarray()
x_non_spam_test_data = vectorizer.transform(x_non_spam_val_data).toarray()

# Convert the spam emails to TF-IDF features for evaluation
train = vectorizer.transform(train).toarray()
spam_emails_df = vectorizer.transform(spam_emails_df).toarray()

# Convert validation test and test data to TF-IDF features
test_data = vectorizer.transform(test).toarray()

# Convert to PyTorch tensors
x_non_spam_train_tensor = torch.tensor(x_non_spam_train_data, dtype=torch.float32)
x_non_spam_val_tensor = torch.tensor(x_non_spam_test_data, dtype=torch.float32)
x_only_non_spam_data_tensor = torch.tensor(train, dtype=torch.float32)
x_only_spam_data_tensor = torch.tensor(spam_emails_df, dtype=torch.float32)
x_test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

# Parameter setup
num_epochs = 40  
learning_rate = 0.001
encoding_dim = 8 # Size of the latent space representation
batch_size = 32
weight_decay = 0

# Autoencoder setup 
input_dim = x_non_spam_train_tensor.shape[1]
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Intialize losses
train_losses = []
val_losses = []

model.train()
# Training the autoencoder
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x_non_spam_train_tensor)

    # Compute the loss
    loss = criterion(output, x_non_spam_train_tensor)
    loss.backward()

    # Update the weights
    optimizer.step()

    # Accumulate the loss
    train_loss = loss.mean().item()
    train_losses.append(train_loss)

    # Evaluation with validation dataset
    with torch.no_grad():
        reconstrutions = model(x_non_spam_val_tensor)
        errors = torch.mean((reconstrutions - x_non_spam_val_tensor) ** 2, dim=1)

        val_loss = errors.mean().item()
        val_losses.append(val_loss)

    # Print loss every 1 epochs
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

# Evaluate the model on the test set
model.eval()
test_losses = 0
with torch.no_grad():
        output = model(x_test_data_tensor)
        loss = criterion(output, x_test_data_tensor)
        test_losses += loss.mean().item()

avg_test_loss = test_losses / len(x_test_data_tensor)
print("Test Loss:", avg_test_loss)

# Plotting classification report and confusion matrix
# Calculate reconstruction errors
with torch.no_grad():
    val_reconstructions = model(x_non_spam_val_tensor)
    val_reconstruction_errors = torch.mean((val_reconstructions - x_non_spam_val_tensor) ** 2, dim=1).numpy()

# Set a threshold for anomaly detection
threshold = np.percentile(val_reconstruction_errors, 95)  # Example: 95th percentile
# threshold = 0.0013
print(f"Threshold for anomaly detection: {threshold:.4f}")

# Calculate reconstruction errors for test data
with torch.no_grad():
    test_reconstructions = model(x_test_data_tensor)
    test_reconstruction_errors = torch.mean((test_reconstructions - x_test_data_tensor) ** 2, dim=1).numpy()

# Classify test emails as spam or non-spam
y_pred = (test_reconstruction_errors > threshold).astype(int)  # 1 for spam, 0 for non-spam

with torch.no_grad():
    pred_non_spam_val= model(x_non_spam_val_tensor).numpy()
    non_spam_val_rsm = np.sqrt(metrics.mean_squared_error(pred_non_spam_val, x_non_spam_val_tensor))

    pred_non_spam = model(x_only_non_spam_data_tensor).numpy()
    non_spam_rsm = np.sqrt(metrics.mean_squared_error(pred_non_spam, x_only_non_spam_data_tensor))

    pred_spam = model(x_only_spam_data_tensor).numpy()
    spam_rmse = np.sqrt(metrics.mean_squared_error(pred_spam, x_only_spam_data_tensor))

print(f"Insample with non spam emails (RMSE): {non_spam_val_rsm}")
print(f"Out of Sample non spam emails (RMSE): {non_spam_rsm}")
print(f"Spam email Score (RMSE): {spam_rmse}")

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')
if os.path.exists('models/autoencoder_model.pth'):
    os.remove('models/autoencoder_model.pth')
torch.save(model, 'models/autoencoder_model.pth')

# Save the vectorizer
if not os.path.exists('models'):
    os.makedirs('models')
if os.path.exists('models/vectorizer.pkl'):
    os.remove('models/vectorizer.pkl')

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)



# Classification report and confusion matrix
print(confusion_matrix(y_pred, test_labels))
print("Accuracy : ", accuracy_score(y_pred, test_labels))
print("Precision : ", precision_score(y_pred ,test_labels, average = 'weighted'))
print("Recall : ", recall_score(y_pred, test_labels, average = 'weighted'))

# Plotting the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses,  label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
if not os.path.exists('graphs'):
    os.makedirs('graphs')
if  os.path.exists('graphs/training_validation_loss.png'):
    os.remove('graphs/training_validation_loss.png')
plt.savefig('graphs/training_validation_loss.png')
plt.show()

# Plotting the reconstruction errors and threshold with validation data
plt.figure(figsize=(10, 6))
plt.hist(val_reconstruction_errors, bins=50, color='lavender', edgecolor='black')
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
plt.axvspan(threshold, max(val_reconstruction_errors), color='red', alpha=0.3)
plt.title('Distribution of Reconstruction Errors with validation data' + '\n' + f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error')
plt.ylabel('Number of Samples')
plt.legend()
plt.grid(True)
plt.tight_layout()
if os.path.exists('graphs/reconstruction_errors_validation.png'):
    os.remove('graphs/reconstruction_errors_validation.png')
plt.savefig('graphs/reconstruction_errors_validation.png')
plt.show()

# Plotting the reconstruction errors and threshold with test data
plt.figure(figsize=(10, 6))
plt.hist(test_reconstruction_errors, bins=50, color='lavender', edgecolor='black')
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
plt.axvspan(threshold, max(test_reconstruction_errors), color='red', alpha=0.3)
plt.title('Distribution of Reconstruction Error with test data' + '\n' + f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error')
plt.ylabel('Number of Samples')
plt.legend()
plt.grid(True)
plt.tight_layout()
if os.path.exists('graphs/reconstruction_errors_test.png'):
    os.remove('graphs/reconstruction_errors_test.png')
plt.savefig('graphs/reconstruction_errors_test.png')
plt.show()

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
TN, FP, FN, TP = cm.ravel()  # Extract values from the confusion matrix
plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
if os.path.exists('graphs/confusion_matrix.png'):
    os.remove('graphs/confusion_matrix.png')
plt.savefig('graphs/confusion_matrix.png')
plt.show() 

# Reconstruction error distribution for spam and non-spam emails
plt.figure(figsize=(10, 6))
plt.hist(test_reconstruction_errors[test_labels == 0], bins=50, color='blue', alpha=0.5, label='Non-Spam')
plt.hist(test_reconstruction_errors[test_labels == 1], bins=50, color='red', alpha=0.5, label='Spam')
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
plt.axvspan(threshold, max(test_reconstruction_errors), color='red', alpha=0.3)
plt.title('Reconstruction Error Distribution for Spam and Non-Spam Emails' + '\n' + f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error')
plt.ylabel('Number of Samples')
plt.legend()
plt.grid()
plt.tight_layout()
if os.path.exists('graphs/reconstruction_error_distribution.png'):
    os.remove('graphs/reconstruction_error_distribution.png')   
plt.savefig('graphs/reconstruction_error_distribution.png')
plt.show()

# Classification report
print(classification_report(test_labels, y_pred, target_names=['Non-Spam', 'Spam'])) 


