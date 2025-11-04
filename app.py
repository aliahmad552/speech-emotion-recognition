from fastapi import FastAPI
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from fastapi.responses import JSONResponses
from tensorflow.keras.models import Sequential

app = FastAPI()

model = tf.keras.models.load_model("best_model.keras")

MODEL_VERSION = '1.11.0'

def preprocess_feature(file):
    y, sr = librosa.load(file)
    mel_spectrogram = librosa.feature.mel_spectrogram(y = y, sr = sr, duration = 4)
    signal = librosa.power_to_db(mel_spectrogram)
    
    prediction = model.predict(signal)
    return prediction

@app.get('/')
def home():
    return {'message':'Speech Emotion Recognition'}

@app.get('/health')
def health():
    status = 'OK'
    model = model is not None

@app.post("/predict")
def predict():
    audio = preprocess_audio(audio)
    prediction = model.predict(audio)

    prediction =np.argmax(prediction,axis = 1)
    return prediction