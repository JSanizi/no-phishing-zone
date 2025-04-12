import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
import torch.nn as nn

def load_data():
    # Import dataset phishing_email.csv
    df = pd.read_csv('datasets/phishing_email.csv')

    """Data cleaning and preprocessing"""
    # Converting 'text_combined' to  lowercase
    df['text_combined'] = df['text_combined'].str.lower()

    # Removing missing values
    df.dropna(inplace=True)

    # Removing duplicates
    df.drop_duplicates(inplace=True)

    # Removing undefined values
    df = df[df['text_combined'] != 'undefined']

    # Decontracting the words
    def decontract(phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    text = decontract(df['text_combined'][70])

    # Replacing numbers
    df['text_combined']=df['text_combined'].str.replace(r'\d+(\.\d+)?', 'numbers')
    #CONVRTING EVERYTHING TO LOWERCASE
    df['text_combined']=df['text_combined'].str.lower()
    #REPLACING NEXT LINES BY 'WHITE SPACE'
    df['text_combined']=df['text_combined'].str.replace(r'\n'," ") 
    # REPLACING EMAIL IDs BY 'MAILID'
    df['text_combined']=df['text_combined'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','MailID')
    # REPLACING URLs  BY 'Links'
    df['text_combined']=df['text_combined'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','Links')
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df['text_combined']=df['text_combined'].str.replace(r'£|\$', 'Money')
    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df['text_combined']=df['text_combined'].str.replace(r'\s+', ' ')
    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df['text_combined']=df['text_combined'].str.replace(r'^\s+|\s+?$', '')
    #REPLACING CONTACT NUMBERS
    df['text_combined']=df['text_combined'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','contact number')
    #REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 
    df['text_combined']=df['text_combined'].str.replace(r"[^a-zA-Z0-9]+", " ")

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['text_combined'] = df['text_combined'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Separating the dataset into non phishing and phishing emails
    X_non_phishing = df[df['label'] == 0]['text_combined']
    X_phishing = df[df['label'] == 1]['text_combined']
    X_all_emails = df['text_combined']


    # Print amount of phishing and non phishing emails
    print("Number of non phishing emails: ", len(X_non_phishing))
    print("Number of phishing emails: ", len(X_phishing))
    print("Total number of emails: ", len(X_all_emails))

    # Text Vectorization
    vectorizer = TfidfVectorizer(max_features=56930)
    X_non_phishing_vectorized = vectorizer.fit_transform(X_non_phishing)
    X_phishing_vectorized = vectorizer.transform(X_phishing)
    X_all_emails_vectorized = vectorizer.transform(X_all_emails)

    return X_non_phishing_vectorized, X_phishing_vectorized, X_all_emails_vectorized
