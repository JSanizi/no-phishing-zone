import os
from dotenv import load_dotenv
import imaplib
import email
from email.header import decode_header
import re

# Load environment variables from .env file
load_dotenv("credential.env")

# Function to connect to the email server and fetch emails
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

# Check if the environment variables are set
if not EMAIL or not PASSWORD:
    raise ValueError("EMAIL and PASSWORD environment variables must be set.")

def connect_to_email():
    """Connects to the Gmail IMAP server and returns the mail object."""
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")
    return mail

def fetch_email_text(mail, email_id):
    """Fetches only the text body from an email, removing links and attachments."""
    status, msg_data = mail.fetch(email_id, "(RFC822)")
    email_content = ""

    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])

            # Extract the email body
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    # Only take the plain text part (skip attachments and HTML)
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        try:
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            email_content += body + "\n"
                        except:
                            email_content += "Unable to decode message content\n"
            else:
                # If not multipart, directly extract the plain text
                email_content = msg.get_payload(decode=True).decode(errors="ignore")

            # Remove links
            email_content = re.sub(r"http\S+|www\S+|https\S+", "", email_content, flags=re.MULTILINE)

            return email_content.strip()  # Return just the text content

def get_latest_email():
    """Fetches the latest unread email and returns its details."""
    mail = connect_to_email()
    status, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()

    if not email_ids:
        print("📭 No unread emails found.")
        mail.close()
        mail.logout()
        return None

    latest_email = fetch_email_text(mail, email_ids[-1])

    mail.close()
    mail.logout()

    return latest_email