# Explainable Audio Deepfake Detection using CNN

> Detect AI-generated audio with deep learning — and understand *why* the model thinks so.

---

## Overview

This project classifies audio clips as **real** (human-generated) or **fake** (AI-generated) using a Convolutional Neural Network (CNN). Audio signals are converted into **Mel spectrograms**, which serve as visual representations that the CNN learns to distinguish.

Beyond raw predictions, the system emphasizes **explainability** — providing confidence scores and spectrogram visualizations so users can interpret and trust the model's decisions.

---

## Project Structure

```
audio-deepfake-detector/
├── data/               # ASVspoof dataset (subset)
├── notebooks/
│   └── setup.ipynb     # Data loading and preprocessing
├── src/                # Model and data pipeline (to be implemented)
├── ui/                 # Streamlit UI (to be implemented)
├── results/            # Outputs and visualizations
├── docs/               # Diagrams and documentation
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone <your-repo-link>
cd audio-deepfake-detector
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

Launch the setup notebook:
```bash
jupyter notebook
```

Then open `notebooks/setup.ipynb`. This notebook:
- Loads the ASVspoof dataset and parses labels (`bonafide` → real, `spoof` → fake)
- Generates Mel spectrograms from raw `.flac` audio
- Visualizes audio waveforms and spectrograms

---

## Dataset

This project uses the **[ASVspoof 2019 Logical Access (LA)](https://www.asvspoof.org/)** dataset.

| Property | Details |
|----------|---------|
| Format | `.flac` audio files |
| Labels | `bonafide` (real), `spoof` (fake) |
| Subset | Reduced subset used for feasibility |

---

## Roadmap

- [x] Data loading and preprocessing
- [x] Mel spectrogram generation
- [ ] CNN model implementation (`src/`)
- [ ] Explainability layer (confidence scores, Grad-CAM)
- [ ] Streamlit UI (`ui/`)

---

## Author

**Sharath Subramanian**
📧 [sharath.talursub@ufl.edu](mailto:sharath.talursub@ufl.edu)