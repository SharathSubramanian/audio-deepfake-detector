import os
import torch
import numpy as np
import json
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve

from src.models import CNN, CNN_Dropout, CNN_Attention
from src.inference import load_model, audio_to_mel


# =========================
# PATHS (VERIFY THESE)
# =========================
DATA_PATH = "data/ASVspoof2019_LA/ASVspoof2019_LA_train/flac"
PROTOCOL_PATH = "data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"


# =========================
# LOAD LABELS
# =========================
def load_labels():
    labels = {}

    with open(PROTOCOL_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            file_id = parts[1]
            label = parts[-1]

            labels[file_id] = 1 if label == "spoof" else 0

    return labels


# =========================
# EER CALCULATION
# =========================
def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer


# =========================
# EVALUATE MODEL
# =========================
def evaluate_model(model, labels):

    y_true = []
    y_pred = []
    y_scores = []

    # 🔥 BALANCED SAMPLING (FIXES YOUR ERROR)
    real_files = [k for k, v in labels.items() if v == 0]
    fake_files = [k for k, v in labels.items() if v == 1]

    n = 250  # increase later if needed

    selected_files = real_files[:n] + fake_files[:n]
    random.shuffle(selected_files)

    for file_id in selected_files:

        file_path = os.path.join(DATA_PATH, file_id + ".flac")

        if not os.path.exists(file_path):
            continue

        try:
            mel = audio_to_mel(file_path)
            tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]

            pred = int(np.argmax(probs))
            score = probs[1]  # fake probability

            y_true.append(labels[file_id])
            y_pred.append(pred)
            y_scores.append(score)

        except Exception as e:
            print(f"Skipping {file_id}: {e}")
            continue

    if len(y_true) == 0:
        raise ValueError("No valid samples processed. Check dataset paths.")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    eer = compute_eer(y_true, y_scores)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "eer": float(eer)
    }


# =========================
# MAIN
# =========================
def main():

    print("Loading labels...")
    labels = load_labels()
    print(f"Total labels loaded: {len(labels)}")

    print("Loading models...")

    models = {
        "CNN": load_model("models/cnn.pth", CNN),
        "Dropout": load_model("models/cnn_dropout.pth", CNN_Dropout),
        "Attention": load_model("models/cnn_attention.pth", CNN_Attention),
    }

    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        try:
            metrics = evaluate_model(model, labels)
            results[name] = metrics
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = "FAILED"

    print("\nFinal Results:")
    print(json.dumps(results, indent=4))

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved to evaluation_results.json")


if __name__ == "__main__":
    main()