# inference.py
import numpy as np
import os
from model_training.model_training import extract_frames, build_resnet50_model, build_lstm_model, build_capsule_network, build_hybrid_model
from tensorflow.keras.models import load_model # type: ignore

IMG_SIZE = 112
SEQ_LENGTH = 20
MODEL_PATH = os.path.join("model_training", "my_model.h5")

# Build architecture
resnet_model = build_resnet50_model((IMG_SIZE, IMG_SIZE, 3))
lstm_model = build_lstm_model((SEQ_LENGTH, 2048))
capsule_model = build_capsule_network((SEQ_LENGTH, 2048))
hybrid_model = build_hybrid_model(resnet_model, lstm_model, capsule_model)

# Load weights
hybrid_model.load_weights(MODEL_PATH)

def analyze_video(video_path):
    frames = extract_frames(video_path)
    input_data = np.expand_dims(frames, axis=0)
    prediction = hybrid_model.predict(input_data)[0]
    fake_score = float(prediction[1]) * 100
    return {"authenticity": round(fake_score, 2)}
