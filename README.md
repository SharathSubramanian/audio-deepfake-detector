# Explainable Audio Deepfake Detection System

> An end-to-end deep learning system for detecting AI-generated audio with explainability, real-time monitoring, and production-ready deployment.

---

## Overview

The rapid advancement of generative AI has made synthetic audio nearly indistinguishable from real human speech, creating serious risks in security, fraud, and misinformation. This project presents a complete, production-oriented machine learning system that not only detects audio deepfakes but explains its decisions and monitors performance in real time.

The system integrates the full ML lifecycle — from data preprocessing and model training to interactive deployment and operational observability — making it suitable for both research and real-world application.

**Core capabilities:**
- Deep learning classification using CNN-based architectures
- Explainable AI via Grad-CAM spectrogram visualizations
- Real-time MLOps monitoring with Prometheus and Grafana
- Interactive web interface built with Streamlit
- Containerized, reproducible deployment via Docker

---

## System Architecture

Audio Input (.wav / .flac)
|
Preprocessing (Librosa)
|
Mel Spectrogram
|
CNN Model (PyTorch)
/        
Prediction   Grad-CAM
\        /
Streamlit UI
|
Prometheus --> Grafana


---

## Models

Three CNN-based architectures were implemented and evaluated:

| Model | Description |
|---|---|
| CNN | Baseline convolutional classifier |
| CNN + Dropout | Regularized variant for improved generalization |
| CNN + Attention | Attention mechanism for enhanced feature focus and interpretability |

---

## Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to the final convolutional layer to produce heatmaps that highlight which regions of the Mel spectrogram most influenced the model's prediction. This provides:

- Visual evidence for each classification decision
- A basis for debugging and model validation
- Improved transparency for non-technical stakeholders

---

## Performance

Evaluated on a subset of the **ASVspoof 2019 Logical Access (LA)** dataset:

| Model | Accuracy | Precision | Recall | F1 Score | EER |
|---|---|---|---|---|---|
| CNN | 97.6% | 0.976 | 0.976 | 0.976 | ~0.00 |
| CNN + Dropout | 97.6% | 0.976 | 0.976 | 0.976 | ~0.00 |
| CNN + Attention | 96.6% | 0.967 | 0.966 | 0.967 | ~0.00 |

> **Note:** Results are based on a representative subset of the dataset. Performance may vary under real-world conditions, particularly with noisy or out-of-distribution audio.

---

## User Interface

The Streamlit interface supports:

- Upload of `.wav` or `.flac` audio files with in-browser playback
- Real-time prediction with confidence score display
- Side-by-side Mel spectrogram and Grad-CAM heatmap visualization
- Model selection (CNN, CNN + Dropout, CNN + Attention)
- Downloadable PDF report per prediction

---

## Monitoring

A production-style observability pipeline is included:

**Prometheus** collects:
- Total prediction count
- Inference latency (per-request histogram)
- Error rate
- Confidence score distribution
- Real vs. fake prediction distribution

**Grafana** visualizes:
- Real-time prediction throughput
- Latency trends over time
- Model usage and class distribution
- Error spikes and anomalies

---

## Project Structure


audio-deepfake-detector/
├── data/                  # Dataset directory (not included, see Dataset section)
├── notebooks/             # Training, experimentation, and preprocessing notebooks
├── src/                   # Core ML pipeline (preprocessing, inference, Grad-CAM)
├── models/                # CNN architecture definitions
├── ui/                    # Streamlit application
├── monitoring/            # Prometheus and Grafana configuration
├── results/               # Evaluation outputs and visualizations
├── evaluate.py            # Batch evaluation script
├── docker-compose.yml     # Full-stack container orchestration
├── Dockerfile             # Application container definition
├── requirements.txt       # Python dependencies
└── README.md


---

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/SharathSubramanian/audio-deepfake-detector.git
cd audio-deepfake-detector
docker-compose up --build
```

Once running, access the services at:

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| Metrics API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### Local Setup

```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## Usage

**Running inference:**
1. Launch the application using Docker or locally
2. Upload a `.wav` or `.flac` audio file
3. Select a model variant
4. View the prediction, confidence score, and Grad-CAM visualization
5. Download the generated PDF report if needed

**Running batch evaluation:**
```bash
python evaluate.py
```
Results are saved to `results/evaluation_results.json`.

---

## Dataset

This project uses the **ASVspoof 2019 Logical Access (LA)** dataset.

- **Labels:** `bonafide` (genuine speech) and `spoof` (synthesized or converted speech)
- **Format:** FLAC audio files at 16 kHz
- **Access:** Available at [https://www.asvspoof.org](https://www.asvspoof.org)

The dataset is not included in this repository due to size constraints. Download it separately and place it in the `data/` directory following the structure expected by the preprocessing pipeline.

---

## Limitations

- Trained and evaluated on a subset of the ASVspoof 2019 dataset; generalization to unseen synthesis techniques may be limited
- Binary classification only (real vs. fake); the specific synthesis algorithm is not identified
- Performance degrades on short clips (under ~1 second) and high-noise recordings
- Grad-CAM heatmaps require some familiarity with spectrograms to interpret meaningfully

---

## Future Work

- Transformer-based architectures (Wav2Vec 2.0, HuBERT) for improved feature representation
- Multi-class classification to identify specific synthesis algorithms
- Evaluation on larger and more diverse datasets (FakeAVCeleb, In-the-Wild)
- Real-time streaming inference for live audio monitoring
- Cloud deployment via AWS SageMaker or Google Vertex AI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | PyTorch |
| Audio Processing | Librosa |
| Explainability | Grad-CAM |
| User Interface | Streamlit |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker, docker-compose |

---

## Author

**Sharath Subramanian**  
University of Florida  
sharath.talursub@ufl.edu  
[github.com/SharathSubramanian](https://github.com/SharathSubramanian)