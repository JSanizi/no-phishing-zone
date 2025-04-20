import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def load_data():
    # Importing SpamAssasin splited dataset
    train_df = pd.read_csv('dataprocess/SpamAssasin/train_non_spam_emails.csv')
    # Importing the test and validation datasets
    val_df = pd.read_csv('dataprocess/SpamAssasin/val_df.csv')
    test_df = pd.read_csv('dataprocess/SpamAssasin/test_df.csv')

    """Data cleaning anxd preprocessing"""
    # Converting the text in subject to lowercase
    train_df['train_subject'] = train_df['subject'].str.lower()
    train_df['train_body'] = train_df['body'].str.lower()

    val_df['val_subject'] = val_df['subject'].str.lower()
    val_df['val_body'] = val_df['body'].str.lower()

    test_df['test_subject'] = test_df['subject'].str.lower()
    test_df['test_body'] = test_df['body'].str.lower()

    # Removing the missing values
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Removing duplicates
    train_df.drop_duplicates(inplace=True)
    val_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)

    # Removing undefined values
    train_df = train_df[train_df['subject'] != 'undefined']
    train_df = train_df[train_df['body'] != 'undefined']

    val_df = val_df[val_df['subject'] != 'undefined']
    val_df = val_df[val_df['body'] != 'undefined']

    test_df = test_df[test_df['subject'] != 'undefined']
    test_df = test_df[test_df['body'] != 'undefined']

    # Combining the subject and body columns
    train_df['sub_body'] = train_df['subject'] +  train_df['body']
    val_df['sub_body'] = val_df['subject'] +  val_df['body']
    test_df['sub_body'] = test_df['subject'] +  test_df['body']

    # Dropping the subject and body columns
    train_df.drop(['subject'], axis=1, inplace=True)
    train_df.drop(['body'], axis=1, inplace=True)

    val_df.drop(['subject'], axis=1, inplace=True)
    val_df.drop(['body'], axis=1, inplace=True)

    test_df.drop(['subject'], axis=1, inplace=True)
    test_df.drop(['body'], axis=1, inplace=True)

    # Decontracting the words
    def decontract(phrase):
        # specific
        phrase = re.sub(r" won\'t ", " will not ", phrase)
        phrase = re.sub(r" can\'t ", " can not ", phrase)

        # general
        phrase = re.sub(r" n\'t ", " not ", phrase)
        phrase = re.sub(r" \'re ", " are ", phrase)
        phrase = re.sub(r" \'s ", " is ", phrase)
        phrase = re.sub(r" \'d ", " would " , phrase)
        phrase = re.sub(r" \'ll ", " will ", phrase)
        phrase = re.sub(r" \'t ", " not ", phrase)
        phrase = re.sub(r" \'ve ", " have ", phrase)
        phrase = re.sub(r" \'m ", " am ", phrase)
        return phrase

    train_df['sub_body'] = train_df['sub_body'].apply(lambda x: decontract(x))
    val_df['sub_body'] = val_df['sub_body'].apply(lambda x: decontract(x))
    test_df['sub_body'] = test_df['sub_body'].apply(lambda x: decontract(x))

    # Replacing numbers
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'\d+(\.\d+)?', ' numbers ')
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'\d+(\.\d+)?', ' numbers ')
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'\d+(\.\d+)?', ' numbers ')

    #CONVRTING EVERYTHING TO LOWERCASE
    train_df['sub_body'] = train_df['sub_body'].str.lower()
    val_df['sub_body'] = val_df['sub_body'].str.lower()
    test_df['sub_body'] = test_df['sub_body'].str.lower()

    #REPLACING NEXT LINES BY 'WHITE SPACE'
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'\n',"  ") 
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'\n'," ")
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'\n',"  ")

    # REPLACING EMAIL IDs BY 'MAILID'
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' MailID ')
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' MailID ')
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' MailID ')

    # REPLACING URLs  BY 'Links'
    train_df['sub_body'] = train_df['sub_body'].str.replace(r' ^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$ ',' Links ')
    val_df['sub_body'] = val_df['sub_body'].str.replace(r' ^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$ ',' Links ')
    test_df['sub_body'] = test_df['sub_body'].str.replace(r' ^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$ ',' Links ')

    # REPLACING CURRENCY SIGNS BY 'MONEY'
    train_df['sub_body'] = train_df['sub_body'].str.replace(r' £|\$ ', ' Money ')
    val_df['sub_body'] = val_df['sub_body'].str.replace(r' £|\$ ', ' Money ')
    test_df['sub_body'] = test_df['sub_body'].str.replace(r' £|\$ ', ' Money ')

    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'\s+', ' ')
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'\s+', ' ')
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'\s+', ' ')

    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'^\s+|\s+?$', '')
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'^\s+|\s+?$', '')
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'^\s+|\s+?$', '')

    #REPLACING CONTACT NUMBERS
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' contact number ')
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' contact number ')
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' contact number ')

    #REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 
    train_df['sub_body'] = train_df['sub_body'].str.replace(r"[^a-zA-Z0-9]+", " ")
    val_df['sub_body'] = val_df['sub_body'].str.replace(r"[^a-zA-Z0-9]+", " ")
    test_df['sub_body'] = test_df['sub_body'].str.replace(r"[^a-zA-Z0-9]+", " ")

    # Remove HTML tags
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'<.*?>', '', regex=True)
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'<.*?>', '', regex=True)
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'<.*?>', '', regex=True)

    # Remove non-alphabetic characters (keep words only)
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    # Replace http links with 'Links'
    train_df['sub_body'] = train_df['sub_body'].str.replace(r'http\S+|www\S+|https\S+', ' Links ', case=False, regex=True)
    val_df['sub_body'] = val_df['sub_body'].str.replace(r'http\S+|www\S+|https\S+', ' Links ', case=False, regex=True)
    test_df['sub_body'] = test_df['sub_body'].str.replace(r'http\S+|www\S+|https\S+', ' Links ', case=False, regex=True)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    train_df['clean_sub_body'] = train_df['sub_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    val_df['clean_sub_body'] = val_df['sub_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    test_df['clean_sub_body'] = test_df['sub_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # Remove features that are not useful
    train_df.drop(['date','label','sender','receiver'], axis=1, inplace=True)
    val_df.drop(['date','label','sender','receiver'], axis=1, inplace=True)
    test_df.drop(['date','label','sender','receiver'], axis=1, inplace=True)

    # Assigning the cleaned data to a new variable
    cleaned_x_train_non_spam = train_df['clean_sub_body']
    cleaned_x_val = val_df['clean_sub_body']
    cleaned_x_test = test_df['clean_sub_body']

    # Printing the cleaned data
    print(cleaned_x_train_non_spam.head())
    print(cleaned_x_val.head())
    print(cleaned_x_test.head())
    
    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train_non_spam_vectorized = vectorizer.fit_transform(cleaned_x_train_non_spam)
    X_val_vectorized = vectorizer.transform(cleaned_x_val)
    X_test_vectorized = vectorizer.transform(cleaned_x_test)

    print(f"Shape of the vectorized data: {X_train_non_spam_vectorized.shape}")
    print(f"Shape of the validation data: {X_val_vectorized.shape}")
    print(f"Shape of the test data: {X_test_vectorized.shape}")

    return X_train_non_spam_vectorized.toarray(), X_val_vectorized.toarray(), X_test_vectorized.toarray()

load_data()