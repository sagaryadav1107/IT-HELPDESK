"""
AI-Based IT Helpdesk Ticket Classification - Model Training Script
Trains an NLP Pipeline (TF-IDF Vectorizer + Multinomial Classifier)
and serializes the trained artifact to 'ticket_classifier_model.pkl'.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Domain Training Corpus ---
DATA = [
    # Network Issue
    ("VPN connection dropping frequently and cannot connect to corporate intranet", "Network Issue"),
    ("Slow internet speed across the entire office floor", "Network Issue"),
    ("Cannot connect to office Wi-Fi network with corporate credentials", "Network Issue"),
    ("DNS failure resolving internal hostname, website unreachable", "Network Issue"),
    ("Ethernet port at desk 4B has no link light or connectivity", "Network Issue"),
    ("Firewall is blocking outbound connections to port 443", "Network Issue"),
    ("Packet loss and high ping latency during video conferences", "Network Issue"),
    ("Network printer IP address cannot be reached or pinged", "Network Issue"),
    ("Subnet mask configuration error on local workstation", "Network Issue"),
    ("Router gateway timeout error when browsing external websites", "Network Issue"),

    # Software Issue
    ("Microsoft Excel crashes every time I open large financial spreadsheets", "Software Issue"),
    ("Outlook not syncing emails, stuck on updating inbox", "Software Issue"),
    ("ERP software displaying license expired error dialog", "Software Issue"),
    ("Zoom meeting client crashing during screen sharing session", "Software Issue"),
    ("Google Chrome browser crashing on startup with memory error", "Software Issue"),
    ("Python runtime missing required dynamic link library on Windows", "Software Issue"),
    ("Antivirus software blocking internal accounting application execution", "Software Issue"),
    ("PDF reader displaying corrupted fonts and unreadable text", "Software Issue"),
    ("Visual Studio Code freezing whenever opening large workspace", "Software Issue"),
    ("Accounting software throws null pointer exception during invoice generation", "Software Issue"),

    # Hardware Issue
    ("External monitor display is completely black and won't turn on", "Hardware Issue"),
    ("Laptop keyboard keys 'E' and 'R' are physically stuck and not responding", "Hardware Issue"),
    ("Mouse cursor is jumping erratically across screen, trackpad broken", "Hardware Issue"),
    ("Laptop battery draining in 15 minutes, charging cable loose", "Hardware Issue"),
    ("Office printer is producing severe paper jams and grinding noise", "Hardware Issue"),
    ("Desktop PC fan is making loud buzzing noises and overheating", "Hardware Issue"),
    ("USB hub ports are not detecting any inserted flash drives or webcams", "Hardware Issue"),
    ("Headphone audio jack is broken and sound only plays in one ear", "Hardware Issue"),
    ("Barcode scanner hardware not transmitting scanned digits", "Hardware Issue"),
    ("Workstation power supply unit smells burned and machine will not power on", "Hardware Issue"),

    # Login/Authentication
    ("My user account is locked after typing the wrong password three times", "Login/Authentication"),
    ("Unable to login to corporate SSO single sign on portal", "Login/Authentication"),
    ("MFA authenticator app token expired and not sending push notifications", "Login/Authentication"),
    ("Forgot my domain password and need an immediate password reset", "Login/Authentication"),
    ("Active Directory user account disabled by administrator", "Login/Authentication"),
    ("Smart card login failure with invalid certificate prompt", "Login/Authentication"),
    ("Temporary login passcode received via SMS has expired", "Login/Authentication"),
    ("Password reset link received in email throws 403 Forbidden error", "Login/Authentication"),
    ("Biometric fingerprint scanner not recognizing employee profile on login", "Login/Authentication"),
    ("Cannot authenticate to VPN server due to invalid client certificate", "Login/Authentication"),

    # System Issue
    ("Production database server down with out of disk space error", "System Issue"),
    ("Linux server experiencing 100% CPU utilization and unresponsiveness", "System Issue"),
    ("Storage Area Network volume unmounted and unavailable to cluster", "System Issue"),
    ("Nightly backup cron job failed with database write lock timeout", "System Issue"),
    ("Permission denied error accessing shared network file storage drive", "System Issue"),
    ("Virtual machine host crashed and rebooted unexpectedly", "System Issue"),
    ("Active Directory domain controller synchronization latency is critical", "System Issue"),
    ("Docker container orchestration service constantly restarting nodes", "System Issue"),
    ("Blue Screen of Death BSOD kernel error code CRITICAL_PROCESS_DIED", "System Issue"),
    ("Web server nginx returning 502 Bad Gateway error on customer portal", "System Issue"),

    # Other
    ("Requesting authorization for two additional computer monitors for desk", "Other"),
    ("Inquiry regarding company IT hardware refresh and upgrade policy", "Other"),
    ("Need guest Wi-Fi access voucher code for visiting client", "Other"),
    ("Requesting software license approval for Adobe Illustrator", "Other"),
    ("General question regarding mobile phone reimbursement procedures", "Other"),
    ("Requesting ergonomic standing desk converter from IT facilities", "Other"),
    ("Where can I find documentation for the new expense reimbursement portal", "Other"),
    ("Need disposal assistance for old decommissioned CRT monitors", "Other"),
]

def train_and_export():
    print("=" * 60)
    print("  Training IT Helpdesk Ticket Classification Model")
    print("=" * 60)

    # 1. Unpack corpus
    X = [item[0] for item in DATA]
    y = [item[1] for item in DATA]

    print(f"[+] Loaded {len(X)} training samples across {len(set(y))} categories.")

    # 2. Build Pipeline (TF-IDF Vectorizer + Multi-class Logistic Classifier)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=5.0,
            max_iter=500,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 3. Train on data
    pipeline.fit(X, y)
    print("[+] Model pipeline training completed successfully.")

    # 4. Evaluate on training corpus
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"[+] Training Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y, y_pred))

    # 5. Serialize model to disk
    output_path = 'ticket_classifier_model.pkl'
    joblib.dump(pipeline, output_path)
    print(f"[+] Serialized model pipeline exported to: '{output_path}'")
    print("=" * 60)

if __name__ == "__main__":
    train_and_export()
