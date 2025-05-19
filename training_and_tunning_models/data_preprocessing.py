import re
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def clean_data(input_data):
    stop_words = set(stopwords.words('english'))

    # If input is a CSV file path
    if isinstance(input_data, str) and input_data.endswith('.csv'):
        df = pd.read_csv(input_data)

        # Converting the text in subject and body to lowercase
        df['subject'] = df['subject'].str.lower()
        df['body'] = df['body'].str.lower()

        # Removing the missing values
        df.dropna(inplace=True)

        # Removing duplicates
        df.drop_duplicates(inplace=True)

        # Removing undefined values
        df = df[df['subject'] != 'undefined']

        # Combining the subject and body columns
        df['sub_body'] = df['subject'] + df['body']

    # If input is a single email string, wrap it in a list
    elif isinstance(input_data, str):
        df = pd.DataFrame({'subject': [''], 'body': [input_data]})
        df['sub_body'] = df['subject'] + df['body']

    # If input is a list of emails, create a DataFrame
    elif isinstance(input_data, list):
        df = pd.DataFrame({'subject': [''] * len(input_data), 'body': input_data})
        df['sub_body'] = df['subject'] + df['body']
    
    # If input is a DataFrame, assume it has 'subject' and 'body' columns
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
        if 'subject' in df.columns and 'body' in df.columns:
            df['sub_body'] = df['subject'] + df['body']
            df['sub_body'] = df['sub_body'].astype(str)
        else:
            raise ValueError("DataFrame must contain 'subject' and 'body' columns.")
    else:   
        raise ValueError("Input must be a CSV file path, a string, or a list of strings.")

    # Decontracting the words
    def decontract(phrase):
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    df['sub_body'] = df['sub_body'].apply(decontract)
    df['sub_body'] = df['sub_body'].str.replace(r'\d+(\.\d+)?', 'numbers', regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r'\n', " ", regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'MailID', regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'Links', regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r'£|\$', 'Money', regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r'\s+', ' ', regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r'^\s+|\s+?$', '', regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'contact number', regex=True)
    df['sub_body'] = df['sub_body'].str.replace(r"[^a-zA-Z0-9]+", " ", regex=True)

    # Remove stopwords
    df['clean_sub_body'] = df['sub_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # For dataset, return features and labels
    if 'label' in df.columns:
        X_non_spam = df[df['label'] == 0]['clean_sub_body']
        X_spam = df[df['label'] == 1]['clean_sub_body']
        X_all_emails = df['clean_sub_body']
        X_labels = df['label']

        print(f"Number of non-spam bodys: {len(X_non_spam)}")
        print(f"Number of spam bodys: {len(X_spam)}")
        print(f"Number of all bodys: {len(X_all_emails)} \n")

        return X_all_emails, X_labels

    # For prediction, just return the cleaned text(s)
    return df['clean_sub_body'].tolist()

def only_tokenize_data(df):
    # Importing the dataset Ling-Spam dataset from kaggle
    df = pd.read_csv(df)

    """Data cleaning and preprocessing"""
    # Converting the text in subject to lowercase
    df['subject'] = df['subject'].str.lower()
    df['body'] = df['body'].str.lower()

    # Removing the missing values
    df.dropna(inplace=True)

    # Removing duplicates
    df.drop_duplicates(inplace=True)

    # Removing undefined values
    df = df[df['subject'] != 'undefined']

    # Combining the subject and body columns
    df['sub_body'] = df['subject'] +  df['body']

    # Dropping the subject and body columns
    df.drop(['subject'], axis=1, inplace=True)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['non_clean_sub_body'] = df['sub_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # Remove features that are not useful
    if 'date' in df.columns:
        df.drop(['date'], axis=1, inplace=True)
    if 'sender' in df.columns:
        df.drop(['sender'], axis=1, inplace=True)
    if 'receiver' in df.columns:
        df.drop(['receiver'], axis=1, inplace=True) 
    else:
        pass

    non_cleaned_data = df['non_clean_sub_body']
    X_labels = df['label']

    return non_cleaned_data, X_labels