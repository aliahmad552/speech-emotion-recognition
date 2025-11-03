from fastapi import FastAPI
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from fastapi.responses import JSONResponses
from tensorflow.keras.models import Sequential

app = FastAPI()

def preprocess_feature(file):
    y, sr = librosa.load(file)
    mel_spectrogram = librosa.feature.mel_spectrogram(y = y, sr = sr, duration = 4)
    