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
import pickle

def load_data():
    # Importing the dataset Ling-Spam dataset from kaggle
    df = pd.read_csv('datasets/CEAS_08.csv')

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

    body=decontract(df['body'][70])

    # Replacing numbers
    df['sub_body']=df['sub_body'].str.replace(r'\d+(\.\d+)?', 'numbers')
    #CONVRTING EVERYTHING TO LOWERCASE
    df['sub_body']=df['sub_body'].str.lower()
    #REPLACING NEXT LINES BY 'WHITE SPACE'
    df['sub_body']=df['sub_body'].str.replace(r'\n'," ") 
    # REPLACING EMAIL IDs BY 'MAILID'
    df['sub_body']=df['sub_body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','MailID')
    # REPLACING URLs  BY 'Links'
    df['sub_body']=df['sub_body'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','Links')
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df['sub_body']=df['sub_body'].str.replace(r'£|\$', 'Money')
    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df['sub_body']=df['sub_body'].str.replace(r'\s+', ' ')
    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df['sub_body']=df['sub_body'].str.replace(r'^\s+|\s+?$', '')
    #REPLACING CONTACT NUMBERS
    df['sub_body']=df['sub_body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','contact number')
    #REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 
    df['sub_body']=df['sub_body'].str.replace(r"[^a-zA-Z0-9]+", " ")

    #CONVRTING EVERYTHING TO LOWERCASE
    df['body']=df['body'].str.lower()
    #REPLACING NEXT LINES BY 'WHITE SPACE'
    df['body']=df['body'].str.replace(r'\n'," ") 
    # REPLACING EMAIL IDs BY 'MAILID'
    df['body']=df['body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','MailID')
    # REPLACING URLs  BY 'Links'
    df['body']=df['body'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','Links')
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df['body']=df['body'].str.replace(r'£|\$', 'Money')
    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df['body']=df['body'].str.replace(r'\s+', ' ')
    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df['body']=df['body'].str.replace(r'^\s+|\s+?$', '')
    #REPLACING CONTACT NUMBERS
    df['body']=df['body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','contact number')
    #REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 
    df['body']=df['body'].str.replace(r"[^a-zA-Z0-9]+", " ")

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['clean_sub_body'] = df['sub_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # Remove features that are not useful
    df.drop(['sub_body', 'body'], axis=1, inplace=True)

    # Separating the dataset into non-spam and spam
    X_non_spam = df[df['label'] == 0]['clean_sub_body']
    X_spam = df[df['label'] == 1]['clean_sub_body']
    X_all_emails = df['clean_sub_body']
    X_labels = df['label']


    # Print amount of non-spam and spam bodys
    print(f"Number of non-spam bodys: {len(X_non_spam)}")
    print(f"Number of spam bodys: {len(X_spam)}")
    print(f"Number of all bodys: {len(X_all_emails)}")

    return X_all_emails, X_labels