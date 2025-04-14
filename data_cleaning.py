import pandas as pd
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords


def clean_data(df):

    if 'subject' and 'body' in df.columns:
        # Converting the text in subject and body to lowercase
        df['subject'] = df['subject'].str.lower()
        df['body'] = df['body'].str.lower()

        # Removing undefined values
        df = df[df['subject'] != 'undefined']
        df = df[df['body'] != 'undefined']

        # Combining the subject and body columns
        df['sub_body'] = df['subject'] + df['body']

    else:
        df['sub_body'] = df


    # Removing the missing values
    df.dropna(inplace=True)

    # Removing duplicates
    df.drop_duplicates(inplace=True)

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
    
    df['sub_body'] = df['sub_body'].apply(decontract)

     # Replacing numbers
    df['sub_body'] = df['sub_body'].str.replace(r'\d+(\.\d+)?', ' numbers ')

    #CONVRTING EVERYTHING TO LOWERCASE
    df['sub_body'] = df['sub_body'].str.lower()

    #REPLACING NEXT LINES BY 'WHITE SPACE'
    df['sub_body'] = df['sub_body'].str.replace(r'\n'," ") 

    # REPLACING EMAIL IDs BY 'MAILID'
    df['sub_body'] = df['sub_body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' MailID ')

    # REPLACING URLs  BY 'Links'
    df['sub_body'] = df['sub_body'].str.replace(r' ^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$ ',' Links ')
   
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df['sub_body'] = df['sub_body'].str.replace(r' £|\$ ', ' Money ')

    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df['sub_body'] = df['sub_body'].str.replace(r'\s+', ' ')
   
    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df['sub_body'] = df['sub_body'].str.replace(r'^\s+|\s+?$', '')
    
    #REPLACING CONTACT NUMBERS
    df['sub_body'] = df['sub_body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',' contact number ')
    
    #REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 
    df['sub_body'] = df['sub_body'].str.replace(r"[^a-zA-Z0-9]+", " ")

    # Remove HTML tags
    df['sub_body'] = df['sub_body'].str.replace(r'<.*?>', '', regex=True)
    
    # Remove non-alphabetic characters (keep words only)
    df['sub_body'] = df['sub_body'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
   
    # Replace http links with 'Links'
    df['sub_body'] = df['sub_body'].str.replace(r'http\S+|www\S+|https\S+', ' Links ', case=False, regex=True)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['clean_sub_body'] = df['sub_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))  

    # Get the labels from columns
    df_labels = get_labels(df)    
    
    # Remove features that are not useful
    # if the dataset has the columns 'date', 'label', 'sender', 'receiver', remove them
    if 'date' in df.columns:
        df.drop(['date'], axis=1, inplace=True)
    if 'sender' in df.columns:
        df.drop(['sender'], axis=1, inplace=True)
    if 'receiver' in df.columns:
        df.drop(['receiver'], axis=1, inplace=True) 
    else:
        pass

    # Assigning the cleaned data to the original dataframe
    cleaned_df = df['clean_sub_body']

    return cleaned_df, df_labels

def get_labels(df):

    # Assigning the labels to the original dataframe
    if 'label' in df.columns:
        df_labels = df['label']
        df.drop(['label'], axis=1, inplace=True)
    else:
        df_labels = None

    return df_labels




