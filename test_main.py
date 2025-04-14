from data_cleaning import clean_data
import pandas as pd

# Load the dataset
df = pd.read_csv('dataprocess/SpamAssasin/train_non_spam_emails.csv')

# Check if the dataset is loaded correctly
print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print("Dataset columns:", df.head)

# Clean the data
X_train = clean_data(df)

# Print the cleaned data
print(X_train.shape)
print(X_train)