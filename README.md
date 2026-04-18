# Audio Deepfake Detection System

> End-to-end deep learning pipeline for detecting AI-generated audio — with explainability, monitoring, and production-ready deployment.

---

## Overview

As synthetic audio becomes increasingly realistic, detecting deepfakes is critical for **security, media integrity, and trust**.  

This project presents a **complete machine learning system** that goes beyond model training to include:

- Explainable predictions (Grad-CAM)
- Real-time monitoring (Prometheus + Grafana)
- Interactive user interface (Streamlit)
- Containerized deployment (Docker)

Audio inputs are converted into **Mel spectrograms** and classified using CNN-based models. Each prediction is accompanied by **visual explanations**, making the system interpretable and trustworthy.

---

## What’s New in Deliverable 3 🚀

This version significantly improves upon the initial prototype:

- Improved preprocessing (normalization, fixed-length inputs)
- More stable model inference and Grad-CAM handling
- Enhanced Streamlit UI with structured outputs
- Real-time monitoring using Prometheus + Grafana
- Extended evaluation with ROC curves and confusion matrices
- Modular, reproducible system design

---

## Key Features

### Deepfake Detection Models

| Model | Description |
|---|---|
| CNN | Baseline convolutional classifier |
| CNN + Dropout | Improved generalization |
| CNN + Attention | Focused feature learning |

---

### Explainability (Grad-CAM)

- Visualizes **where the model is looking**
- Highlights spectrogram regions influencing predictions
- Improves transparency and debugging

---

### Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score  
- Equal Error Rate (EER)  
- ROC Curve and Confusion Matrix  

Results are saved in `evaluation_results.json`.

---

### Interactive UI (Streamlit)

- Upload or record audio
- View predictions and confidence
- Visualize spectrogram + Grad-CAM
- Compare multiple models
- Download automated PDF report

---

### Monitoring (MLOps)

- Tracks predictions, latency, and errors
- Real-time dashboards via Grafana
- Production-style observability pipeline

---

## Architecture

```
Audio Input (.wav / .flac)
        ↓
Mel Spectrogram (librosa)
        ↓
CNN Model (PyTorch)
        ↓
Prediction + Grad-CAM
        ↓
Streamlit UI + PDF Report
        ↓
Prometheus → Grafana Monitoring
```

---

## Model Performance

Evaluated on **ASVspoof 2019 LA dataset (subset)**:

| Model | Accuracy | Precision | Recall | F1 | EER |
|---|---|---|---|---|---|
| CNN | 97.6% | 0.976 | 0.976 | 0.976 | 0.000 |
| CNN + Dropout | 97.6% | 0.976 | 0.976 | 0.976 | 0.000 |
| CNN + Attention | 96.6% | 0.967 | 0.966 | 0.967 | 0.000 |

> Note: Results are based on a subset and may not generalize to all real-world audio conditions.

---

## Project Structure

```
audio-deepfake-detector/
│
├── data/                # Dataset (not included)
├── notebooks/           # Training, testing, evaluation
├── src/                 # Core ML pipeline
├── ui/                  # Streamlit app
├── monitoring/          # Prometheus config
├── results/             # Outputs and visualizations
├── models/              # Model definitions
│
├── evaluate.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start (Docker)

```bash
git clone https://github.com/SharathSubramanian/audio-deepfake-detector.git
cd audio-deepfake-detector
docker-compose up --build
```

### Access Services

| Service | URL |
|---|---|
| UI | http://localhost:8501 |
| Metrics | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## Usage

### Run Inference

- Upload `.wav` or `.flac`
- View prediction + explanation
- Download PDF report

---

### Run Evaluation

```bash
python evaluate.py
```

---

## Dataset

- **ASVspoof 2019 Logical Access (LA)**
- Labels: `bonafide` vs `spoof`
- Not included due to size

---

## Limitations ⚠️

- Trained on a limited dataset subset
- Performance may drop on noisy audio
- Binary classification only (no spoof type detection)
- Requires careful interpretation of Grad-CAM

---

## Future Work

- Larger and more diverse datasets
- Transformer-based models
- Multi-class spoof detection
- Cloud deployment

---

## Tech Stack

| Category | Tools |
|---|---|
| ML | PyTorch |
| Audio | Librosa |
| UI | Streamlit |
| Explainability | Grad-CAM |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker |

---

## What Makes This Project Stand Out

- Full ML pipeline (not just a model)
- Explainable AI integration
- Real-time monitoring (MLOps)
- Interactive UI
- Production-style deployment

---

## Author

Sharath Subramanian  
Email: sharath.talursub@ufl.edu  
GitHub: https://github.com/SharathSubramanian
