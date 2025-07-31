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
    """
    Send email alert for suspicious activities or walking deviations.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine alert type and customize message
    if "walking_deviation" in suspicious_category.lower():
        alert_type = "Walking Deviation Detected"
        priority = get_deviation_priority(suspicious_category)
        subject = f"[{priority}] {alert_type} - {suspicious_category.replace('_', ' ').title()}"
        
        body = f"""
Walking Deviation Alert!

Deviation Type : {suspicious_category.replace('_', ' ').title()}
Person ID      : {person_id}
Camera         : {camera_id}
Time           : {now}
Priority       : {priority}

Clinical Analysis Summary:
- This alert is based on weighted clinical gait analysis
- Metrics analyzed: Leg Asymmetry, Step Consistency, Lateral Stability, Joint Movement, Vertical Oscillation
- Recommended action: {get_recommended_action(suspicious_category)}

Please check the attached screenshot for visual confirmation.

Note: This system uses clinical literature-based thresholds for gait analysis.
For medical concerns, please consult healthcare professionals.
        """
    else:
        # Original suspicious activity alert
        alert_type = "Suspicious Activity Detected"
        subject = f"[ALERT] {alert_type} - {suspicious_category}"
        
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

    # Attach screenshot if available
    if screenshot_path and os.path.exists(screenshot_path):
        with open(screenshot_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(screenshot_path)
        msg.add_attachment(file_data, maintype='image', subtype='png', filename=file_name)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ssl.create_default_context()) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"[EMAIL SENT] {alert_type} alert to {to_emails}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

def get_deviation_priority(deviation_category):
    """Determine priority level based on deviation severity."""
    if "severe" in deviation_category.lower():
        return "HIGH PRIORITY"
    elif "moderate" in deviation_category.lower():
        return "MEDIUM PRIORITY"
    elif "mild" in deviation_category.lower():
        return "LOW PRIORITY"
    else:
        return "INFO"

def get_recommended_action(deviation_category):
    """Get recommended action based on deviation type."""
    if "severe" in deviation_category.lower():
        return "Immediate medical attention may be required. Consider emergency response."
    elif "moderate" in deviation_category.lower():
        return "Medical evaluation recommended. Monitor closely."
    elif "mild" in deviation_category.lower():
        return "Continue monitoring. Document patterns for medical review."
    else:
        return "Standard monitoring protocol."