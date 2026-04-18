def audio_to_mel(path, apply_augmentation=False):
    y, _ = librosa.load(path, sr=SR)

    # =========================
    # NEW: Trim silence
    # =========================
    y, _ = librosa.effects.trim(y)

    # =========================
    # KEEP ORIGINAL LENGTH LOGIC
    # =========================
    max_len = SR * 3
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    if apply_augmentation:
        noise = np.random.normal(0, 0.003, len(y))
        y = y + noise

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    mel = cv2.resize(mel, (128, 128))

    return mel