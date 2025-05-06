import os
import random
import json
import numpy as np # type: ignore
import cv2 # type: ignore
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, GlobalAveragePooling2D, Reshape, TimeDistributed, Concatenate, Conv1D, MaxPooling1D, Flatten # type: ignore
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Constants
IMG_SIZE = 112
SEQ_LENGTH = 20
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'deepfake dataset'))
TRAIN_FOLDER = os.path.join(DATASET_PATH, "train_sample_videos")
TEST_FOLDER = os.path.join(DATASET_PATH, "test_videos")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'model_training', 'my_model.h5')

def extract_frames(video_path, max_frames=SEQ_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        normalized = frame / 255.0
        frames.append(normalized)
        frame_count += 1

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Path does not exist - {folder_path}")
        return False
    if not os.listdir(folder_path):
        print(f"Warning: Folder is empty - {folder_path}")
        return False
    return True

def load_labels(metadata_path):
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return {}
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def build_resnet50_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dense(64, activation='relu')(x)
    return Model(inputs, x)

def build_capsule_network(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return Model(inputs, x)

def build_hybrid_model(resnet_model, lstm_model, capsule_model):
    video_input = Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    reshaped = Reshape((SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))(video_input)
    features = TimeDistributed(resnet_model)(reshaped)
    lstm_features = lstm_model(features)
    capsule_features = capsule_model(features)
    combined = Concatenate()([lstm_features, capsule_features])
    x = Dense(64, activation='relu')(combined)
    output = Dense(2, activation='softmax')(x)
    return Model(inputs=video_input, outputs=output)

def plot_history(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Validate dataset folder
    if not check_folder(TRAIN_FOLDER):
        raise ValueError("Training folder is invalid or empty.")

    # Load metadata for labels
    metadata_path = os.path.join(TRAIN_FOLDER, 'metadata.json')
    metadata = load_labels(metadata_path)
    if not metadata:
        raise ValueError("Failed to load metadata for labels.")

    # Get video files
    video_files = [f for f in os.listdir(TRAIN_FOLDER) if f.endswith((".mp4", ".avi", ".mov"))]
    if not video_files:
        raise ValueError("No video files found in training folder!")

    # Sample videos (use at least 20 videos if available, or all if fewer)
    num_selected_files = min(max(20, len(video_files) // 5), len(video_files))
    selected_video_files = random.sample(video_files, num_selected_files)

    X, y = [], []
    for video_file in selected_video_files:
        video_path = os.path.join(TRAIN_FOLDER, video_file)
        try:
            frames = extract_frames(video_path)
            X.append(frames)
            # Assign label based on metadata
            label = 1 if metadata.get(video_file, {}).get('label') == 'FAKE' else 0
            y.append(label)
            print(f"Processed {video_file} with label {label}")
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

    if not X:
        raise ValueError("No valid frames extracted from any video.")

    # Prepare data
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=2)

    # Split into train and test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) < 2:
        raise ValueError("Not enough training samples after splitting. Need at least 2 samples.")

    # Build the model
    resnet_model = build_resnet50_model((IMG_SIZE, IMG_SIZE, 3))
    lstm_model = build_lstm_model((SEQ_LENGTH, 2048))
    capsule_model = build_capsule_network((SEQ_LENGTH, 2048))
    hybrid_model = build_hybrid_model(resnet_model, lstm_model, capsule_model)

    hybrid_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    # Define checkpoint to save the best model directly to MODEL_SAVE_PATH
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

    # Train the model
    history = hybrid_model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=4,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )

    # Plot training history
    plot_history(history)
    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()