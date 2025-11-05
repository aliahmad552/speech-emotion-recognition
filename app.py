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
EMOTIONS = ['anger', 'calm', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
# ============================

app = FastAPI(title="Speech Emotion Recognition API")

# serve templates folder
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    mn = mel_spec.min()
    mx = mel_spec.max()
    if mx - mn == 0:
        norm = np.zeros_like(mel_spec, dtype=np.float32)
    else:
        norm = (mel_spec - mn) / (mx - mn)

    # add channel and batch dims: (1, 128, 345, 1)
    inp = np.expand_dims(norm, axis=(0, -1)).astype(np.float32)
    return inp

# ========== End preprocessing ==========

@app.post("/predict", response_class=JSONResponse)
async def predict(audio_file: UploadFile = File(...)):
    # accept only wav
    if not audio_file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported. Please upload a WAV audio file.")

    # save to a temp file then process because librosa needs a path or file-like object
    try:
        contents = await audio_file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Ensure file is readable by soundfile/librosa (sometimes header issues for uploads)
        # Re-write with soundfile to ensure consistent format
        data, samplerate = sf.read(tmp_path, dtype='float32')
        sf.write(tmp_path, data, SR)

        mel = process_audio(tmp_path)  # uses your function
        x = prepare_input_from_array(mel)
        preds = model.predict(x)  # shape (1, num_classes)
        probs = softmax(preds).numpy().squeeze()  # ensure probabilities
        top_idx = int(np.argmax(probs))
        emotion = EMOTIONS[top_idx]
        confidence = float(probs[top_idx])

        # detailed per-class probabilities
        class_probs = {label: float(probs[i]) for i, label in enumerate(EMOTIONS)}

        return JSONResponse({
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "probs": class_probs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "emotions": EMOTIONS})
