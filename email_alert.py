# email_alert.py
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

def send_email_alert(suspicious_category, person_id, camera_id, screenshot_path, to_emails):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[ALERT] Suspicious Activity Detected - {suspicious_category}"
    body = f"""
Suspicious Activity Detected!

Category   : {suspicious_category}
Person ID  : {person_id}
Camera     : {camera_id}
Time       : {now}

Please check the attached screenshot for more details.
    """

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ', '.join(to_emails)
    msg.set_content(body)

    if screenshot_path:
        with open(screenshot_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(screenshot_path)
        msg.add_attachment(file_data, maintype='image', subtype='png', filename=file_name)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ssl.create_default_context()) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"[EMAIL SENT] Alert to {to_emails}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
