import torch
import numpy as np
import librosa
import torch.nn.functional as F

SR = 16000
N_MELS = 128
FIXED_LENGTH = 128  # IMPORTANT: must match training


# =========================
# AUDIO → MEL SPECTROGRAM
# =========================
import cv2

def audio_to_mel(path):
    y, _ = librosa.load(path, sr=SR)

    max_len = SR * 3
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)

    # ✅ ADD THESE (CRITICAL FIX)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = cv2.resize(mel, (128, 128))

    return mel 


# =========================
# LOAD MODEL
# =========================
def load_model(path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# =========================
# GRAD-CAM (ONLY FOR ATTENTION MODEL)
# =========================
def generate_gradcam(model, class_idx):
    gradients = model.gradients
    feature_maps = model.feature_maps

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1)

    cam = F.relu(cam)
    cam = cam.squeeze().detach().numpy()

    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam


# =========================
# PREDICT
# =========================
def predict(model, audio_path):
    mel = audio_to_mel(audio_path)

    tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

    explain = hasattr(model, "attention")

    if explain:
        output = model(tensor, explain=True)
    else:
        output = model(tensor)

    probs = torch.softmax(output, dim=1).detach().numpy()[0]

    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    cam = None
    if explain:
        model.zero_grad()
        output[0, pred].backward()
        cam = generate_gradcam(model, pred)

    return mel, pred, conf, probs, cam