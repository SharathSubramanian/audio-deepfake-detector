import json
import os
from datetime import datetime

LOG_FILE = "logs/predictions.json"

os.makedirs("logs", exist_ok=True)

def log_prediction(filename, prediction, confidence):

    entry = {
        "timestamp": datetime.now().isoformat(),
        "file": filename,
        "prediction": int(prediction),
        "confidence": float(confidence)
    }

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

    with open(LOG_FILE, "r+") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)