import os
import json
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

def get_credentials():
    creds = None
    token_path = "token.json"
    if os.path.exists(token_path):
        with open(token_path, "r") as token:
            try:
                creds_info = json.load(token)
                creds = Credentials.from_authorized_user_info(creds_info, SCOPES)
            except json.JSONDecodeError:
                creds = None
    if creds and creds.valid and creds.has_scopes(SCOPES):
        return creds
    else:
        if os.path.exists(token_path):
            os.remove(token_path)
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
        return creds

creds = get_credentials()
service = build("gmail", "v1", credentials=creds)

def send_email(mail_results):
    subject = mail_results['subject']
    html_content = mail_results['html_content']
    message = MIMEMultipart()
    message["to"] = os.getenv('SEND_TO')
    message["subject"] = subject
    html_part = MIMEText(html_content, 'html')
    message.attach(html_part)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    message_body = {"raw": raw}
    service.users().messages().send(userId="me", body=message_body).execute()
    print(f"Email sent success!")