import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


model = load_model('model.pkl')


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    return mfcc


le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)

st.title("Speech Recognition using CNN")

audio_file = st.file_uploader("Upload an audio file", type=["wav"])

if audio_file is not None:
    
    features = extract_features(audio_file)
    prediction = model.predict(features)
    predicted_label = le.inverse_transform([np.argmax(prediction)])

    
    st.write(f"Predicted Label: {predicted_label[0]}")
