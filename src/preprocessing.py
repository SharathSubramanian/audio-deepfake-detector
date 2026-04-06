import librosa
import numpy as np

SR = 16000
DURATION = 3

def audio_to_mel(path):
    y, _ = librosa.load(path, sr=SR)

    max_len = SR * DURATION
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)

    return mel