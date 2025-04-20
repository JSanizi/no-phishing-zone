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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset


# Split dataset into train, validation, and test sets
train_df, val_df, test_df, spam_emails_df = split_dataset('datasets/SpamAssasin.csv')

# Clean the data
train, train_labels = clean_data(train_df)
val, val_labels = clean_data(val_df)
test, test_labels = clean_data(test_df)

# Split train data into train and validation sets
x_non_spam_train_data, x_non_spam_test_data = train_test_split(train, test_size=0.25, random_state=42)

# Convert the data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
x_non_spam_train_data = vectorizer.fit_transform(x_non_spam_train_data).toarray()
x_non_spam_test_data = vectorizer.transform(x_non_spam_test_data).toarray()
train = vectorizer.transform(train).toarray()
spam_emails_df = vectorizer.transform(spam_emails_df).toarray()

# Convert to PyTorch tensors
x_non_spam_train_tensor = torch.tensor(x_non_spam_train_data, dtype=torch.float32)
x_non_spam_test_tensor = torch.tensor(x_non_spam_test_data, dtype=torch.float32)
x_only_non_spam_data_tensor = torch.tensor(train, dtype=torch.float32)
x_only_spam_data_tensor = torch.tensor(spam_emails_df, dtype=torch.float32)

# Parameter setup
num_epochs = 40
learning_rate = 0.001
encoding_dim = 32  # Size of the latent space representation
batch_size = 8
weight_decay = 1e-5

# Autoencoder setup 
input_dim = x_non_spam_train_tensor.shape[1]
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    # Print loss every 100 epochs
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation with only non-spam dataset.
model.eval()
with torch.no_grad():
    pred_test = model(x_non_spam_test_tensor).numpy()
    score1 = np.sqrt(metrics.mean_squared_error(pred_test, x_non_spam_test_data))

    pred_good = model(x_only_non_spam_data_tensor).numpy()
    score2 = np.sqrt(metrics.mean_squared_error(pred_good, x_only_non_spam_data_tensor))

    pred_bad = model(x_only_spam_data_tensor).numpy()
    score3 = np.sqrt(metrics.mean_squared_error(pred_bad, x_only_spam_data_tensor))

print(f"Insample with non spam emails (RMSE): {score1}")
print(f"Out of Sample non spam emails (RMSE): {score2}")
print(f"Spam email Score (RMSE): {score3}")


# Plotting the results using classification report and confusion matrix
def plot_classification_report(y_true, y_pred, title='Classification Report'):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, fmt='.2f', cmap='Blues')
    plt.title(title)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()
