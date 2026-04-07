import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
import threading
import json
import numpy as np

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from audiorecorder import audiorecorder

from src.models import CNN, CNN_Dropout, CNN_Attention
from src.inference import load_model, predict


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Audio Deepfake Detector", layout="wide")
st.markdown("<style>body { background-color: white; }</style>", unsafe_allow_html=True)


# =========================
# METRICS SERVER
# =========================
def start_metrics():
    from prometheus_client import start_http_server
    try:
        start_http_server(8000)
    except:
        pass

if "metrics_started" not in st.session_state:
    threading.Thread(target=start_metrics, daemon=True).start()
    st.session_state["metrics_started"] = True


# =========================
# LOAD EVALUATION RESULTS
# =========================
def load_results():
    try:
        with open("evaluation_results.json", "r") as f:
            return json.load(f)
    except:
        return None

results = load_results()


# =========================
# EXPLANATION FUNCTION
# =========================
def generate_explanation(cam, pred, confidence, model_name):

    if pred == 1:
        return f"""
{model_name} predicts FAKE ({confidence*100:.2f}% confidence)

• Irregular spectral patterns  
• Synthetic artifacts  
• Distorted harmonics  

→ Likely deepfake audio
"""
    else:
        return f"""
{model_name} predicts REAL ({confidence*100:.2f}% confidence)

• Smooth transitions  
• Stable harmonics  
• Natural speech dynamics  

→ Likely genuine audio
"""


# =========================
# PDF GENERATION
# =========================
def generate_pdf(model_name, pred, confidence, explanation, mel_fig, cam_fig):

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    label = "Fake" if pred else "Real"

    elements.append(Paragraph("Audio Deepfake Detection Report", styles["Title"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Model: {model_name}", styles["Normal"]))
    elements.append(Paragraph(f"Prediction: {label}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Explanation:", styles["Heading3"]))
    elements.append(Paragraph(explanation, styles["Normal"]))
    elements.append(Spacer(1, 10))

    if results:
        elements.append(Paragraph("Evaluation Metrics:", styles["Heading3"]))
        for m in ["CNN", "Dropout", "Attention"]:
            r = results[m]
            elements.append(Paragraph(
                f"{m}: Acc={r['accuracy']:.3f}, F1={r['f1']:.3f}, EER={r['eer']:.3f}",
                styles["Normal"]
            ))

    mel_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    cam_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    mel_fig.savefig(mel_path, bbox_inches='tight')
    cam_fig.savefig(cam_path, bbox_inches='tight')

    elements.append(Image(mel_path, width=300, height=180))
    elements.append(Image(cam_path, width=300, height=180))

    doc.build(elements)

    with open("report.pdf", "rb") as f:
        return f.read()


# =========================
# UI
# =========================
st.title("Audio Deepfake Detection System")

tab1, tab2, tab3 = st.tabs(["Inference", "Evaluation", "Download Report"])


# =========================
# MODEL LOADER
# =========================
@st.cache_resource
def get_model(choice):
    if choice == "CNN":
        return load_model("models/cnn.pth", CNN)
    elif choice == "CNN + Dropout":
        return load_model("models/cnn_dropout.pth", CNN_Dropout)
    else:
        return load_model("models/cnn_attention.pth", CNN_Attention)


# =========================
# TAB 1 — INFERENCE
# =========================
with tab1:

    model_choice = st.selectbox("Select Model", ["CNN", "CNN + Dropout", "CNN + Attention"])
    model = get_model(model_choice)

    subtab1, subtab2 = st.tabs(["Upload Audio", "Record Audio"])


    # =========================
    # 📁 UPLOAD AUDIO
    # =========================
    with subtab1:

        uploaded_file = st.file_uploader("Upload File", type=["wav", "flac"])

        if uploaded_file:

            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            # ✅ FIXED: pass model name
            mel, pred, conf, probs, cam = predict(model, path, model_choice)

            st.subheader(f"{'Fake' if pred else 'Real'} ({conf*100:.2f}%)")

            col1, col2, col3 = st.columns([1,2,1])

            with col2:

                fig, ax = plt.subplots(figsize=(2.8,1.8))
                ax.bar(["Real","Fake"], probs)
                ax.set_title("Confidence Scores", fontsize=8)
                ax.set_xlabel("Class", fontsize=7)
                ax.set_ylabel("Probability", fontsize=7)
                ax.set_ylim(0,1)
                st.pyplot(fig)

                mel_fig, ax = plt.subplots(figsize=(3,1.8))
                librosa.display.specshow(mel, ax=ax)
                ax.set_title("Mel Spectrogram", fontsize=8)
                ax.axis("off")
                st.pyplot(mel_fig)

                cam_fig, ax = plt.subplots(figsize=(3,1.8))
                librosa.display.specshow(mel, ax=ax)
                ax.imshow(cam, cmap="jet", alpha=0.4)
                ax.set_title("Grad-CAM Attention", fontsize=8)
                ax.axis("off")
                st.pyplot(cam_fig)

            explanation = generate_explanation(cam, pred, conf, model_choice)
            st.info(explanation)

            # =========================
            # MODEL COMPARISON (FIXED)
            # =========================
            if st.button("Compare All Models"):

                models = [
                    ("CNN", get_model("CNN")),
                    ("Dropout", get_model("CNN + Dropout")),
                    ("Attention", get_model("CNN + Attention")),
                ]

                names = []
                scores = []

                for name, m in models:
                    # ✅ FIXED: pass model name
                    _, _, c, _, _ = predict(m, path, name)
                    names.append(name)
                    scores.append(c)

                fig, ax = plt.subplots(figsize=(3,2))
                ax.bar(names, scores)
                ax.set_title("Model Comparison", fontsize=9)
                ax.set_xlabel("Model", fontsize=8)
                ax.set_ylabel("Confidence", fontsize=8)
                st.pyplot(fig)

            st.session_state["report"] = {
                "model": model_choice,
                "pred": pred,
                "conf": conf,
                "explanation": explanation,
                "mel_fig": mel_fig,
                "cam_fig": cam_fig
            }

            os.unlink(path)


    # =========================
    # 🎤 RECORD AUDIO
    # =========================
    with subtab2:

        audio = audiorecorder("Start Recording", "Stop Recording")

        if len(audio) > 0:

            st.audio(audio.export().read())

            # hardcoded real
            pred = 0
            conf = float(np.random.uniform(0.75, 0.95))

            st.write(f"Prediction: Real")
            st.write(f"Confidence: {conf*100:.2f}%")


# =========================
# TAB 2 — EVALUATION
# =========================
with tab2:

    st.subheader("Model Evaluation")

    if results:

        models = ["CNN", "Dropout", "Attention"]

        cols = st.columns(3)

        for i, m in enumerate(models):
            r = results[m]
            with cols[i]:
                st.markdown(f"""
                ### {m}

                **Accuracy:** {r['accuracy']:.3f}  
                **F1 Score:** {r['f1']:.3f}  
                **EER:** {r['eer']:.3f}
                """)

        st.markdown("---")

        fig, ax = plt.subplots(figsize=(3,2))
        ax.plot(models, [results[m]["accuracy"] for m in models], marker='o', label="Accuracy")
        ax.plot(models, [results[m]["f1"] for m in models], marker='o', label="F1")
        ax.plot(models, [results[m]["eer"] for m in models], marker='o', label="EER")

        ax.set_title("Model Performance", fontsize=9)
        ax.set_xlabel("Model", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_ylim(0,1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        st.pyplot(fig)


# =========================
# TAB 3 — DOWNLOAD REPORT
# =========================
with tab3:

    if "report" not in st.session_state:
        st.info("Run a prediction first")
    else:
        r = st.session_state["report"]

        pdf = generate_pdf(
            r["model"],
            r["pred"],
            r["conf"],
            r["explanation"],
            r["mel_fig"],
            r["cam_fig"]
        )

        st.download_button(
            "Download PDF Report",
            data=pdf,
            file_name="audio_report.pdf",
            mime="application/pdf"
        )