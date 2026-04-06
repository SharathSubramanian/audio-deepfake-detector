from src.models import CNN_Attention
from src.inference import load_model, predict

MODEL_PATH = "models/cnn_attention.pth"

AUDIO_PATH = "data/ASVspoof2019_LA/ASVspoof2019_LA_train/flac/LA_T_3653923.flac"


def main():
    print("Loading model...")
    model = load_model(MODEL_PATH, CNN_Attention)

    print("Running prediction...")
    mel, pred, conf, probs, cam = predict(model, AUDIO_PATH)

    print("Prediction:", "Fake" if pred == 1 else "Real")
    print("Confidence:", conf)
    print("Probabilities:", probs)

    if cam is not None:
        print("Grad-CAM generated successfully")


if __name__ == "__main__":
    main()