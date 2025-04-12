import os
from dotenv import load_dotenv
import imaplib
import email
import re

# Load environment variables from .env file
load_dotenv("credential.env")

# Function to connect to the email server and fetch emails
EMAIL = os.getenv("EMAIL_RECEIVER")
PASSWORD = os.getenv("PASSWORD_RECEIVER")

# Check if the environment variables are set
if not EMAIL or not PASSWORD:
    raise ValueError("EMAIL and PASSWORD environment variables must be set.")

def connect_to_email():
    """Connects to the Gmail IMAP server and returns the mail object."""
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")  # You can specify a different folder if needed
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

def get_unread_emails():
    """Fetches all unread emails from the inbox."""
    mail = connect_to_email()
    status, messages = mail.search(None, "UNSEEN")  # Look for unread messages

    email_ids = messages[0].split()  # Split the email IDs returned by the search command

    if not email_ids:
        print("📭 No unread emails found.")
        mail.close()
        mail.logout()
        return None
    
    unread_emails = []  # This will store the fetched unread emails

    for email_id in email_ids:
        # Fetch and process each unread email
        email_text = fetch_email_text(mail, email_id)
        
        if email_text:
            unread_emails.append(email_text)

    mail.close()
    mail.logout()

    return unread_emails  # Return a list of unread emails (text content)
