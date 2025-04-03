import re
import string
import torch
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from connect_to_email import get_latest_email

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

    # 5️⃣ Convert to lowercase
    email_text = email_text.lower()

    # 6️⃣ Remove stopwords
    email_text = " ".join([word for word in email_text.split() if word not in stop_words])

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    email_text_vectorized = vectorizer.fit_transform([email_text])

    return email_text_vectorized.toarray()
