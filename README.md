# Audio Deepfake Detection System

> An end-to-end deep learning pipeline for classifying audio as **real** or **AI-generated** — with explainability, production monitoring, and containerized deployment.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps & Monitoring](#mlops--monitoring)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Use Cases](#use-cases)
- [Author](#author)

---

## Overview

As AI-generated audio grows increasingly indistinguishable from genuine speech, reliable detection systems are critical for trust, security, and media integrity. This project presents a **production-grade audio deepfake detection system** that goes beyond model training — incorporating explainability, real-time monitoring, and containerized deployment.

Audio clips are converted into **Mel spectrograms** and classified using CNN-based architectures. Predictions are accompanied by **Grad-CAM visualizations** that highlight the spectrogram regions driving each decision, making the system transparent and auditable.

---

## Key Features

### Deepfake Detection

Three CNN-based model variants, each offering different performance/robustness trade-offs:

| Model | Description |
|---|---|
| `CNN` | Baseline convolutional classifier |
| `CNN + Dropout` | Regularized variant for improved generalization |
| `CNN + Attention` | Attention-augmented model for focused feature learning |

### Explainability (XAI)

- **Grad-CAM** heatmaps overlaid on Mel spectrograms
- Highlights frequency-time regions most influential to the prediction
- Supports model transparency and human-in-the-loop review

### Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- **Equal Error Rate (EER)** — the standard metric for speaker verification and anti-spoofing systems
- Results exported to `evaluation_results.json` for reproducibility

### Interactive UI (Streamlit)

- Upload or simulate audio inference
- Side-by-side model comparison
- Confidence scores and spectrogram/Grad-CAM display
- Clean, professional interface

### Automated PDF Report Generation

Each inference generates a downloadable report containing:
- Prediction result and confidence score
- Model explanation and rationale
- Mel spectrogram and Grad-CAM visualization
- Full evaluation metrics

---

## Architecture

```
Audio Input (.wav / .flac)
        |
        v
+---------------------+
|  Mel Spectrogram    |  <- librosa preprocessing
|  Conversion         |
+----------+----------+
           |
           v
+---------------------+
|  CNN Model          |  <- CNN / CNN+Dropout / CNN+Attention
|  (PyTorch)          |
+----------+----------+
           |
     +-----+------+
     v            v
Prediction    Grad-CAM
(Real/Fake)   Heatmap
     |            |
     +-----+------+
           v
  +---------------+        +------------------+
  |  Streamlit UI |------->|  PDF Report      |
  +-------+-------+        +------------------+
          |
          v
  +---------------+        +------------------+
  |  Prometheus   |------->|  Grafana         |
  |  Metrics API  |        |  Dashboard       |
  +---------------+        +------------------+
```

---

## Model Performance

All models were evaluated on the **ASVspoof 2019 LA** dataset subset.

| Model | Accuracy | Precision | Recall | F1 Score | EER |
|---|---|---|---|---|---|
| CNN (Baseline) | 97.6% | 0.976 | 0.976 | 0.976 | 0.000 |
| CNN + Dropout | 97.6% | 0.976 | 0.976 | 0.976 | 0.000 |
| CNN + Attention | 96.6% | 0.967 | 0.966 | 0.967 | 0.000 |

> **EER = 0.000** across all variants indicates no operating point where false acceptance and false rejection rates are equal — a strong anti-spoofing result.

---

## Project Structure

```
audio-deepfake-detector/
|
+-- data/                        # ASVspoof dataset (subset, not tracked)
|
+-- notebooks/
|   +-- setup.ipynb              # Data loading, preprocessing, EDA
|   +-- test.ipynb               # Model testing
|   +-- train.ipynb              # Model training
|   +-- evaluation.ipynb         # Model evaluation
|
models                           # CNN, CNN+Dropout, CNN+Attention definitions
+-- src/                         # Core ML pipeline
|   +-- inference.py             # Prediction and Grad-CAM generation
|   +-- metrics.py               # Accuracy, F1, EER computation
|   +-- preprocessing.py         # Audio to Mel spectrogram pipeline
|
+-- ui/                          # Streamlit frontend
|   +-- app.py
|
+-- monitoring/                  # Observability stack
|   +-- prometheus.yml           # Prometheus scrape configuration
|
+-- docs/                        # Architecture diagrams and documentation
+-- results/                     # Saved outputs and visualizations
|
+-- evaluate.py                  # Standalone evaluation script
+-- evaluation_results.json      # Persisted evaluation metrics
|
+-- docker-compose.yml           # Multi-service orchestration
+-- Dockerfile                   # Application container definition
+-- requirements.txt             # Python dependencies
+-- README.md
```

---

## Quick Start

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Clone the repository
git clone https://github.com/SharathSubramanian/audio-deepfake-detector.git
cd audio-deepfake-detector

# 2. Launch all services
docker-compose up --build
```

All services will be available at:

| Service | URL | Description |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Interactive inference interface |
| Metrics API | http://localhost:8000 | Prometheus-compatible metrics endpoint |
| Prometheus | http://localhost:9090 | Metrics collection and querying |
| Grafana | http://localhost:3000 | Real-time monitoring dashboards |

---

## Installation

For local development without Docker:

```bash
# 1. Clone the repository
git clone https://github.com/SharathSubramanian/audio-deepfake-detector.git
cd audio-deepfake-detector

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit App

```bash
streamlit run ui/app.py
```

### Run Evaluation

```bash
python evaluate.py
```

Results are written to `evaluation_results.json`.

### Launch the Setup Notebook

```bash
jupyter notebook notebooks/setup.ipynb
```

This notebook walks through dataset loading, label parsing (`bonafide` -> real, `spoof` -> fake), Mel spectrogram generation, and waveform visualization.

---

## MLOps & Monitoring

This project ships with a full observability pipeline — a deliberate design choice to mirror production ML systems.

### Prometheus Metrics

The metrics API (`localhost:8000`) exposes:

| Metric | Description |
|---|---|
| `predictions_total` | Cumulative prediction count |
| `predictions_by_model` | Per-model usage breakdown |
| `prediction_latency_seconds` | Inference latency histogram |
| `confidence_score` | Distribution of model confidence values |
| `prediction_errors_total` | Error count for reliability tracking |

### Grafana Dashboard

Grafana (`localhost:3000`) provides real-time visualization of:

- Predictions per second
- Real vs. Fake classification distribution
- Per-model performance comparison
- Latency trends over time

> **Default credentials:** `admin` / `admin` (change on first login)

---

## Dataset

This project uses the **[ASVspoof 2019 Logical Access (LA)](https://www.asvspoof.org/)** benchmark dataset — the standard evaluation corpus for audio anti-spoofing research.

| Property | Details |
|---|---|
| Format | `.flac` audio files |
| Labels | `bonafide` (genuine), `spoof` (AI-generated) |
| Task | Binary classification: Real vs. Fake |
| Subset Used | Reduced subset for feasibility |

> The dataset is **not included** in this repository. Download instructions are available at [asvspoof.org](https://www.asvspoof.org/).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch |
| Audio Processing | Librosa |
| Explainability | Grad-CAM (OpenCV) |
| Frontend | Streamlit |
| Report Generation | ReportLab / FPDF |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker, Docker Compose |
| Language | Python 3.8+ |

---

## Use Cases

- **Audio forensics** — Verify authenticity of recordings in legal or journalistic contexts
- **Deepfake detection research** — Benchmark and extend CNN-based anti-spoofing approaches
- **AI safety tooling** — Detect synthetic media in automated content pipelines
- **Explainable AI demonstrations** — Showcase Grad-CAM interpretability in a real-world system
- **MLOps education** — Reference implementation of monitoring, evaluation, and deployment patterns

---


## What Makes This Project Stand Out

This system goes beyond a standard model training exercise to demonstrate a **complete ML engineering workflow**:

- End-to-end pipeline from raw audio to prediction
- Explainability via Grad-CAM — not just predictions, but *why*
- Production-style monitoring with Prometheus + Grafana
- Fully containerized with Docker Compose
- Automated PDF report generation per inference
- Clean, interactive Streamlit UI for non-technical users

---

## Author

**Sharath Subramanian**
Email: [sharath.talursub@ufl.edu](mailto:sharath.talursub@ufl.edu)
GitHub: [github.com/SharathSubramanian](https://github.com/SharathSubramanian)

