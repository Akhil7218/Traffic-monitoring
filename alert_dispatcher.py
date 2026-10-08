"""
TrafficSentinel AI — Alert Dispatcher & Multi-Channel Notification Service
Sends automated email (SMTP) and WhatsApp (Meta Graph API) notifications for detected infractions.
"""

import os
import time
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Channel Credentials
WHATSAPP_PHONE_ID = os.environ.get("WA_PHONE_ID", "")
WHATSAPP_TOKEN    = os.environ.get("WA_TOKEN",    "")
WHATSAPP_API_URL  = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"

CITIZEN_WA_NUMBER = os.environ.get("CITIZEN_WA", "")
ADMIN_WA_NUMBER   = os.environ.get("ADMIN_WA",   "")

GMAIL_ADDRESS      = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASS", "")

CITIZEN_EMAIL = os.environ.get("CITIZEN_EMAIL", "")
ADMIN_EMAIL   = os.environ.get("ADMIN_EMAIL",   "")

MAX_RETRIES  = 3
RETRY_DELAY  = 2


def _is_whatsapp_active():
    return bool(WHATSAPP_PHONE_ID and WHATSAPP_TOKEN)


def _is_email_active():
    sanitized_pass = GMAIL_APP_PASSWORD.replace(" ", "")
    return bool(GMAIL_ADDRESS and sanitized_pass)


def _execute_with_retry(task_func, service_label):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if task_func():
                return True
            raise RuntimeError("Task returned False status")
        except Exception as err:
            last_err = err
            if attempt < MAX_RETRIES:
                print(f"  [{service_label}] ⚠️ Attempt {attempt}/{MAX_RETRIES} failed: {err} — retrying in {RETRY_DELAY}s...", flush=True)
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [{service_label}] ❌ All {MAX_RETRIES} attempts failed. Final error: {last_err}", flush=True)
    return False


def dispatch_infraction_alert(violation_id, plate, violation, fine, timestamp, pdf_path,
                              owner_name="Not Available", owner_phone=None, owner_email=None):
    """
    Dispatches automated violation notices via Email and WhatsApp.
    """
    print(f"  [AlertDispatcher] Processing infraction #{violation_id} ({plate} - {violation})")

    # Email Dispatch
    if _is_email_active():
        destination_email = owner_email or CITIZEN_EMAIL or ADMIN_EMAIL
        if destination_email:
            def send_email_job():
                message = MIMEMultipart()
                message['Subject'] = f"Traffic Infraction Notice: TS-{violation_id:06d} ({plate})"
                message['From'] = GMAIL_ADDRESS
                message['To'] = destination_email

                body = (
                    f"TRAFFIC SENTINEL AI — INFRACTION NOTICE\n\n"
                    f"Citation Ref : TS-{violation_id:06d}\n"
                    f"Plate Number : {plate}\n"
                    f"Registered Owner : {owner_name}\n"
                    f"Infraction Category : {violation}\n"
                    f"Timestamp : {timestamp}\n"
                    f"Penalty Fee : Rs. {fine:,}\n\n"
                    f"Official PDF citation attached with evidence snapshot and payment details.\n"
                    f"TrafficSentinel AI Automated Enforcement Engine."
                )
                message.attach(MIMEText(body, 'plain'))

                if os.path.exists(pdf_path):
                    with open(pdf_path, 'rb') as pdf_file:
                        attachment = MIMEApplication(pdf_file.read(), _subtype="pdf")
                        attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
                        message.attach(attachment)

                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp_server:
                    smtp_server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
                    smtp_server.send_message(message)
                return True

            _execute_with_retry(send_email_job, "Email Alert")

    # WhatsApp Dispatch
    if _is_whatsapp_active():
        destination_phone = owner_phone or CITIZEN_WA_NUMBER or ADMIN_WA_NUMBER
        if destination_phone:
            def send_whatsapp_job():
                payload = {
                    "messaging_product": "whatsapp",
                    "to": destination_phone,
                    "type": "text",
                    "text": {
                        "body": (
                            f"🚨 *TRAFFIC SENTINEL AI CITATION*\n\n"
                            f"*Citation:* TS-{violation_id:06d}\n"
                            f"*Plate:* {plate}\n"
                            f"*Owner:* {owner_name}\n"
                            f"*Infraction:* {violation}\n"
                            f"*Time:* {timestamp}\n"
                            f"*Penalty:* Rs. {fine:,}\n\n"
                            f"Settle fine online at parivahan.gov.in"
                        )
                    }
                }
                response = requests.post(
                    WHATSAPP_API_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
                    timeout=10
                )
                return response.status_code == 200

            _execute_with_retry(send_whatsapp_job, "WhatsApp Alert")


def dispatch_daily_summary(statistics):
    """Sends daily executive summary email to system administrator."""
    if not _is_email_active() or not ADMIN_EMAIL:
        return

    def send_summary_job():
        message = MIMEMultipart()
        message['Subject'] = f"TrafficSentinel AI — Daily Summary ({datetime.now().strftime('%d %b %Y')})"
        message['From'] = GMAIL_ADDRESS
        message['To'] = ADMIN_EMAIL

        body = (
            f"TRAFFIC SENTINEL AI — DAILY ENFORCEMENT SUMMARY\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Total Infractions Captured : {statistics.get('total', 0)}\n"
            f"  - No Helmet              : {statistics.get('no_helmet', 0)}\n"
            f"  - Triple Riding          : {statistics.get('triple_riding', 0)}\n"
            f"  - Counterflow / Wrong Way : {statistics.get('wrong_way', 0)}\n\n"
            f"Total Fines Generated      : Rs. {statistics.get('total_fines', 0):,}\n"
            f"Settled Citations          : {statistics.get('paid', 0)}\n"
            f"Pending Citations          : {statistics.get('pending', 0)}\n"
        )
        message.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp_server:
            smtp_server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp_server.send_message(message)
        return True

    _execute_with_retry(send_summary_job, "Daily Summary Email")


# Backward Compatibility Aliases
notify_violation = dispatch_infraction_alert
send_daily_summary = dispatch_daily_summary
