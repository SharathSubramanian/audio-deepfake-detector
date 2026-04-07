import torch
import numpy as np
import librosa
import torch.nn.functional as F
import cv2

SR = 16000
N_MELS = 128
FIXED_LENGTH = 128

import time
from src.metrics import PREDICTIONS, LATENCY, CONFIDENCE, ERRORS
def audio_to_mel(path):
    y, _ = librosa.load(path, sr=SR)

    max_len = SR * 3
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = cv2.resize(mel, (128, 128))

    return mel


def load_model(path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def generate_gradcam(model):

    if hasattr(model, "gradients") and model.gradients is not None:
        gradients = model.gradients
    elif model.feature_maps is not None and model.feature_maps.grad is not None:
        gradients = model.feature_maps.grad
    else:
        return np.zeros((128, 128))  # SAFE fallback

    feature_maps = model.feature_maps

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1)

    cam = F.relu(cam)
    cam = cam.squeeze().detach().numpy()

    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam


import time
from src.metrics import PREDICTIONS, LATENCY, CONFIDENCE, ERRORS

def predict(model, audio_path, model_name="CNN"):

    start = time.time()

    try:
        mel = audio_to_mel(audio_path)

        tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

        output = model(tensor, explain=True)

        probs = torch.softmax(output, dim=1).detach().numpy()[0]

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        # Grad-CAM
        model.zero_grad()
        output[0, pred].backward()
        cam = generate_gradcam(model)

        # ✅ METRICS UPDATE
        label = "fake" if pred == 1 else "real"
        PREDICTIONS.labels(model=model_name, prediction=label).inc()
        CONFIDENCE.observe(conf)

        return mel, pred, conf, probs, cam

    except Exception as e:
        ERRORS.inc()
        raise e

    finally:
        LATENCY.observe(time.time() - start)