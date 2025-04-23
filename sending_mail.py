import smtplib
import pandas as pd
import os
from dotenv import load_dotenv

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env
load_dotenv('credential.env')

# Email credentials
sender_email_login = os.getenv("EMAIL_SENDER")
receiver_email = os.getenv("EMAIL_RECEIVER")
password = os.getenv("PASSWORD_SENDER")

# SMTP server config
smtp_server = "smtp.gmail.com"
smtp_port = 587

# Load your dataset
dataset = pd.read_csv('datasets/CEAS_08.csv')
dataset = dataset.dropna(subset=['subject', 'body'])

# Shuffle the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# ✨ Limit to amount of emails max per run
MAX_EMAILS = 8
emails_sent = 0
email_number = 1
spam_count = 0
non_spam_count = 0

def send_email(subject, plain_body, html_body, spoofed_sender, receiver, label):
    msg = MIMEMultipart('alternative')
    msg['From'] = spoofed_sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.add_header("X-Spam-Label", str(label))

    msg.attach(MIMEText(plain_body, 'plain'))

    if pd.notna(html_body) and html_body.strip() != '':
        msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email_login, password)
            server.sendmail(sender_email_login, receiver, msg.as_string())
            print(f"🌈 Email {email_number} sent as [{spoofed_sender}] to {receiver} — Label: {label}")
    except Exception as e:
        print(f"💔 Error sending email: {e}")
    

# 🌸 Email loop with limit
for index, row in dataset.iterrows():
    if emails_sent >= MAX_EMAILS:
        print("✨ Limit reached☕💖")
        print(f"✨ Total spam emails sent: {spam_count}")
        print(f"✨ Total non-spam emails sent: {non_spam_count}")
        print("✨ Now you can take a break!")
        break

    subject = row.get('subject', 'No subject')
    plain_body = row.get('body', 'No body content')
    html_body = row.get('html_body', '')  # Optional HTML version
    spoofed_sender = row.get('sender', 'unknown@phish.com')
    label = row.get('label', 1)

    send_email(subject, plain_body, html_body, spoofed_sender, receiver_email, label)
    emails_sent += 1
    email_number += 1

    if label == 1:
        spam_count += 1
    else:
        non_spam_count += 1

# Return what spam and non-spam emails were sent in a array
def get_sent_emails_list():
    # Get the sent email list and their labels and save them in a csv file
    sent_email_df = pd.DataFrame({
        'email_number': range(1, emails_sent + 1),
        'true_label': dataset['label'][:emails_sent],
    })
 
    sent_email_df['true_label'] = sent_email_df['true_label'].apply(lambda x: 'Spam' if x == 1 else 'Non-Spam')
    sent_email_df['true_label'] = sent_email_df['true_label'][::-1]

    if os.path.exists('true_sent_emails.csv'):
        os.remove('true_sent_emails.csv')
    sent_email_df.to_csv('true_sent_emails.csv', index=False)

    true_sent_email_list = pd.read_csv('true_sent_emails.csv')

    print(f"The list of true sent emails labels: {sent_email_df.head()}")
    
    return true_sent_email_list

get_sent_emails_list()