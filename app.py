# main.py
import os
import io
import tempfile
from typing import List

import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax

# ========== CONFIG ==========
MODEL_PATH = "model.keras"  # place your model here
SR = 44100
DURATION = 4  # seconds
N_MELS = 128
TARGET_WIDTH = 345  # time frames expected by your model
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
# ============================

app = FastAPI(title="Speech Emotion Recognition API")

# serve templates folder
templates = Jinja2Templates(directory="templates")

# load model on startup
@app.on_event("startup")
def load_ser_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Put your model.keras next to main.py")
    model = load_model(MODEL_PATH)
    # optionally print model summary to logs
    try:
        model.summary()
    except Exception:
        pass


# ========== Your provided preprocessing, slightly extended ==========
def process_audio(path):
    '''
    Load the audio file, convert to mel spectrogram, return as numpy array.
    (Your original function — kept the same sampling rate and padding behaviour)
    '''
    audio, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
    if len(audio) < DURATION * sr:
        audio = np.pad(audio, pad_width=(0, DURATION * sr - len(audio)), mode='constant')
    signal = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    signal = librosa.power_to_db(signal, ref=np.min)
    image = np.array(signal)
    return image

def prepare_input_from_array(mel_spec: np.ndarray) -> np.ndarray:
    """
    Ensure mel_spec has shape (N_MELS, TARGET_WIDTH), pad/crop the time axis,
    add channel dimension and normalize to 0-1.
    Returns array shaped (1, 128, 345, 1)
    """
    # mel_spec shape: (128, time_frames)
    h, w = mel_spec.shape
    if h != N_MELS:
        raise ValueError(f"Expected {N_MELS} mel bands but got {h}")

    # Crop or pad width to TARGET_WIDTH
    if w < TARGET_WIDTH:
        # pad on the right with minimum value (preserve dB scale)
        pad_width = TARGET_WIDTH - w
        pad_values = np.full((h, pad_width), fill_value=np.min(mel_spec), dtype=mel_spec.dtype)
        mel_spec = np.concatenate([mel_spec, pad_values], axis=1)
    elif w > TARGET_WIDTH:
        mel_spec = mel_spec[:, :TARGET_WIDTH]

    # Normalize per-sample to 0-1 to stabilize input (adjust if model expects different scaling)
    mn = -3.7909594
    std = 54.28706

    # Standardize with training mean and std
    norm = (mel_spec - mn) / std
    # add channel and batch dims: (1, 128, 345, 1)
    inp = np.expand_dims(norm, axis=(0, -1)).astype(np.float32)
    return inp

# ========== End preprocessing ==========

@app.post("/predict", response_class=JSONResponse)
async def predict(audio_file: UploadFile = File(...)):
    try:
        contents = await audio_file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        print(f"[DEBUG] Temp path: {tmp_path}")
        mel = process_audio(tmp_path)
        print(f"[DEBUG] Mel shape: {mel.shape}")
        x = prepare_input_from_array(mel)
        print(f"[DEBUG] Input shape to model: {x.shape}, dtype={x.dtype}")

        preds = model.predict(x)
        print(f"[DEBUG] Model raw preds shape: {preds.shape}")

        top_idx = int(np.argmax(preds))
        print(f"[DEBUG] Predicted index: {top_idx}")

        emotion = EMOTIONS[top_idx]
        logits = preds.squeeze()
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        confidence = float(probs[top_idx])

        class_probs = {label: float(probs[i]) for i, label in enumerate(EMOTIONS)}
        return JSONResponse({
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "probs": class_probs
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "emotions": EMOTIONS})
