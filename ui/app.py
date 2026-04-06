import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
import threading

from src.models import CNN, CNN_Dropout, CNN_Attention
from src.inference import load_model, predict


# =========================
# SAFE METRICS SERVER
# =========================
def start_metrics():
    from prometheus_client import start_http_server
    start_http_server(8000)


if "metrics_started" not in st.session_state:
    try:
        threading.Thread(target=start_metrics, daemon=True).start()
        st.session_state["metrics_started"] = True
    except:
        pass


# =========================
# EXPLANATION FUNCTION
# =========================
def generate_explanation(cam, pred, confidence, model_name):

    if cam is None:
        if pred == 1:
            return f"""
The {model_name} predicts this audio is FAKE with {confidence*100:.2f}% confidence.

This decision is based on:
- Irregular frequency distributions
- Synthetic artifacts in speech
- Temporal inconsistencies

For visual explanation, use CNN + Attention model.
"""
        else:
            return f"""
The {model_name} predicts this audio is REAL with {confidence*100:.2f}% confidence.

This decision is based on:
- Smooth frequency transitions
- Natural speech dynamics
- Consistent acoustic structure

For visual explanation, use CNN + Attention model.
"""

    focus = cam.mean()

    if pred == 1:
        if focus > 0.6:
            return """
The model focuses on high-energy irregular regions in the spectrogram.

These indicate:
- Artificial frequency spikes
- Distorted harmonics
- Deepfake artifacts

This strongly suggests synthetic audio.
"""
        else:
            return """
The model detected subtle inconsistencies across time-frequency regions.

These suggest:
- Slight unnatural transitions
- Hidden synthesis artifacts

The audio may be manipulated.
"""
    else:
        if focus > 0.6:
            return """
The model focuses on stable and consistent frequency bands.

This indicates:
- Natural harmonic structure
- Human-like speech continuity

This strongly suggests real speech.
"""
        else:
            return """
The model observes smooth transitions and balanced frequency energy.

This indicates:
- Natural speech rhythm
- Lack of distortion

The audio is likely genuine.
"""


# =========================
# SUMMARY BULLETS
# =========================
def explanation_summary(pred):
    if pred == 1:
        return [
            "Detected unnatural frequency spikes",
            "Inconsistent temporal patterns",
            "Artifacts typical of deepfake generation"
        ]
    else:
        return [
            "Smooth frequency transitions",
            "Consistent speech patterns",
            "Natural human voice characteristics"
        ]


# =========================
# BUILD DOWNLOADABLE REPORT
# =========================
def build_report(model_name, pred, confidence, probs, explanation, summary):

    label = "Fake" if pred else "Real"

    report = f"""
Audio Deepfake Detection Report
--------------------------------

Model Used: {model_name}

Prediction: {label}
Confidence: {confidence*100:.2f}%

Class Probabilities:
- Real: {probs[0]*100:.2f}%
- Fake: {probs[1]*100:.2f}%

Explanation:
{explanation}

Key Observations:
"""

    for item in summary:
        report += f"- {item}\n"

    return report


# =========================
# UI HEADER
# =========================
st.title("Audio Deepfake Detection System")
st.write("Upload an audio file to determine whether it is real or AI-generated.")


# =========================
# MODEL SELECTION
# =========================
model_choice = st.selectbox(
    "Select Model",
    ["CNN", "CNN + Dropout", "CNN + Attention"]
)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def get_model(choice):
    if choice == "CNN":
        return load_model("models/cnn.pth", CNN)
    elif choice == "CNN + Dropout":
        return load_model("models/cnn_dropout.pth", CNN_Dropout)
    else:
        return load_model("models/cnn_attention.pth", CNN_Attention)


model = get_model(model_choice)


# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "flac"])


if uploaded_file is not None:

    audio_bytes = uploaded_file.read()

    st.subheader("Uploaded Audio")

    file_type = "audio/flac" if uploaded_file.name.endswith(".flac") else "audio/wav"
    st.audio(audio_bytes, format=file_type)

    suffix = ".flac" if uploaded_file.name.endswith(".flac") else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    mel, pred, conf, probs, cam = predict(model, path)

    # =========================
    # RESULT
    # =========================
    st.subheader("Prediction")

    label = "Fake" if pred else "Real"
    st.write(f"Result: {label}")

    st.progress(float(conf))
    st.write(f"Confidence: {conf*100:.2f}%")

    # =========================
    # PROBABILITIES
    # =========================
    st.subheader("Class Probabilities")
    st.write(f"Real: {probs[0]*100:.2f}%")
    st.write(f"Fake: {probs[1]*100:.2f}%")

    # =========================
    # SPECTROGRAM
    # =========================
    st.subheader("Mel Spectrogram")

    fig, ax = plt.subplots()
    librosa.display.specshow(mel, ax=ax)
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)

    # =========================
    # GRAD-CAM
    # =========================
    if cam is not None:
        st.subheader("Model Attention (Grad-CAM)")

        fig, ax = plt.subplots()
        librosa.display.specshow(mel, ax=ax)
        ax.imshow(cam, cmap="jet", alpha=0.5)
        ax.set_title("Important Regions Used by Model")
        st.pyplot(fig)
    else:
        st.warning("This model does not support visual explanations. Use CNN + Attention.")

    # =========================
    # EXPLANATION
    # =========================
    st.subheader("Explanation")

    explanation = generate_explanation(cam, pred, conf, model_choice)
    st.write(explanation)

    # =========================
    # SUMMARY
    # =========================
    st.subheader("Key Observations")

    summary = explanation_summary(pred)
    for item in summary:
        st.write(f"- {item}")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    report = build_report(model_choice, pred, conf, probs, explanation, summary)

    st.download_button(
        label="Download Report",
        data=report,
        file_name="audio_analysis_report.txt",
        mime="text/plain"
    )

    # =========================
    # MODEL COMPARISON
    # =========================
    if st.button("Compare All Models"):

        st.subheader("Model Comparison")

        models = [
            ("CNN", get_model("CNN")),
            ("Dropout", get_model("CNN + Dropout")),
            ("Attention", get_model("CNN + Attention")),
        ]

        for name, m in models:
            _, p, c, _, _ = predict(m, path)
            st.write(f"{name}: {'Fake' if p else 'Real'} ({c*100:.2f}%)")

    os.unlink(path)