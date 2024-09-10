import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


model = load_model('voice_classification_model.h5')


def extract_features(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=30)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)

st.title("Speech Recognition using CNN")

audio_file = st.file_uploader("Upload an audio file", type=["wav"])

if audio_file is not None:
    
    features = extract_features(audio_file)
    prediction = model.predict(features)
    predicted_label = le.inverse_transform([np.argmax(prediction)])

    
    st.write(f"Predicted Label: {predicted_label[0]}")
