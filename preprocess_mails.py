import re
import string
import nltk
import pickle

from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def preprocess_email(email_text):
    # 1️⃣ Remove HTML tags
    email_text = BeautifulSoup(email_text, "html.parser").get_text()

    # 2️⃣ Remove URLs
    email_text = re.sub(r"http\S+|www\S+|https\S+", "", email_text, flags=re.MULTILINE)

    # 3️⃣ Remove email addresses
    email_text = re.sub(r"\S*@\S*\s?", "", email_text)

    # 4️⃣ Remove special characters & punctuation
    email_text = email_text.translate(str.maketrans("", "", string.punctuation))

    # 7️⃣ Remove extra whitespace
    email_text = re.sub(r"\s+", " ", email_text).strip()

    # Replacing numbers
    email_text = re.sub(r"\d+(\.\d+)?", "numbers", email_text)

    #CONVRTING EVERYTHING TO LOWERCASE
    email_text = email_text.lower()

    #REPLACING NEXT LINES BY 'WHITE SPACE'
    email_text = email_text.replace("\n", " ")

    # REPLACING CURRENCY SIGNS BY 'MONEY'
    email_text = re.sub(r"£|\$", "Money", email_text)

    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    email_text = re.sub(r"\s+", " ", email_text)

    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    email_text = re.sub(r"^\s+|\s+?$", "", email_text)

    # REPLACING EMAIL IS BY 'MAILID'
    email_text = re.sub(r"^.+@[^\.].*\.[a-z]{2,}$", "MailID", email_text)

    # REPLACING URLs  BY 'Links'
    email_text = re.sub(r"^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$", "Links", email_text)

    # REPLACING CONTACT NUMBERS
    email_text = re.sub(r"^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$", "contact number", email_text)

    # REPLACING SPECIAL CHARACTERS BY WHITE SPACE
    email_text = re.sub(r"[^a-zA-Z0-9]+", " ", email_text)

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

    email_text = decontract(email_text)

    # 6️⃣ Remove stopwords
    email_text = " ".join([word for word in email_text.split() if word not in stop_words])

    # Text Vectorization
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    email_text_vectorized = vectorizer.transform([email_text])

    return email_text_vectorized.toarray()