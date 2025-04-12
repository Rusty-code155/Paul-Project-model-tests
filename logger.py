#Code By Turner Miles Peeples
import csv
import os
from datetime import datetime

def log_feedback(group_id, score, loss_history, phase="training"):
    log_path = f"logs/{phase}_feedback_log.csv"
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(log_path)

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Group ID", "Score", "Loss History"])
        writer.writerow([datetime.now().isoformat(), group_id, score, loss_history])