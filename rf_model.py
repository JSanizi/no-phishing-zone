import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
from dataprocess_nb import load_data

# Random Forest Classifier
tvec = TfidfVectorizer()

# Import the csv file from before
X, Y = load_data()

# Splitting the data into features and labels
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 225,stratify=Y)

rfc = RFC(random_state=42)
model_7 = Pipeline([('vectorizer',tvec),('classifier',rfc)])

model_7.fit(X_train,Y_train)

y_pred = model_7.predict(X_test)
print(confusion_matrix(y_pred,Y_test))
print("Accuracy : ", accuracy_score(y_pred,Y_test))
print("Precision : ", precision_score(y_pred,Y_test, average = 'weighted'))
print("Recall : ", recall_score(y_pred,Y_test, average = 'weighted'))

# Confusion matrix
cm = confusion_matrix(Y_test, y_pred)
TN, FP, FN, TP = cm.ravel()  # Extract values from the confusion matrix
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.4f}")
print(f"True Positives: {TP}, True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}")
print(f"Spam emails detected: {TP}, Non-spam emails detected: {TN}")
print(f"Spam emails misclassified as non-spam: {FN}, Non-spam emails misclassified as spam: {FP}")

plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
if os.path.exists('graphs/rf_confusion_matrix.png'):
    os.remove('graphs/rf_confusion_matrix.png')
plt.savefig('graphs/rf_confusion_matrix.png')
plt.show() 