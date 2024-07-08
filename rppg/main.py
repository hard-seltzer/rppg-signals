import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def preprocess_frame(frame, face_coords=None):
    if face_coords is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None
        
        face_coords = faces[0]
    
    x, y, w, h = face_coords
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    return face, face_coords

def build_masked_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    flattened_input = Flatten()(input_layer)
    encoded = Dense(128, activation='relu')(flattened_input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(decoded)
    decoded = Reshape(input_shape)(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(), loss=MeanSquaredError())
    return autoencoder

def train_autoencoder(autoencoder, frames, epochs=50, batch_size=16):
    autoencoder.fit(frames, frames, epochs=epochs, batch_size=batch_size, shuffle=True)
    return autoencoder

def extract_rppg_from_model(autoencoder, frame):
    decoded_frame = autoencoder.predict(np.expand_dims(frame, axis=0))
    green_channel = decoded_frame[0, :, :, 1]
    avg_green = np.mean(green_channel)
    return avg_green

video_path = "vid.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    plt.figure(figsize=(15, 8))
    face_coords = None
    rppg_signal = []
    frame_count = 0
    start_time = time.time()
    frames_buffer = []
    input_shape = (64, 64, 3)
    autoencoder = build_masked_autoencoder(input_shape)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        processed_face, detected_coords = preprocess_frame(frame, face_coords)
        
        if processed_face is not None:
            if face_coords is None:
                face_coords = detected_coords
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            x, y, w, h = face_coords
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract rPPG signal
            rppg_value = extract_rppg_from_model(autoencoder, processed_face)
            rppg_signal.append(rppg_value)
            
            plt.clf()  # Clear the current figure
            
            # Plot original frame with face detection
            plt.subplot(221)
            plt.imshow(frame_rgb)
            plt.title("Frame with Face Detection")
            plt.axis('off')
            
            # Plot processed face
            plt.subplot(222)
            plt.imshow(processed_face)
            plt.title("Processed Face")
            plt.axis('off')
            
            # Plot rPPG signal
            plt.subplot(212)
            plt.plot(rppg_signal)
            plt.title("rPPG Signal")
            plt.xlabel("Frame")
            plt.ylabel("Green Channel Average")
            
            plt.tight_layout()
            plt.pause(0.01)  # Pause to update the plot
            
            # Calculate and print FPS every second
            if time.time() - start_time >= 1:
                fps = frame_count / (time.time() - start_time)
                print(f"Current FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
            
            # Train the autoencoder
            if len(frames_buffer) == 100:
                autoencoder = train_autoencoder(autoencoder, np.array(frames_buffer), epochs=10, batch_size=16)
                frames_buffer = []
            else:
                frames_buffer.append(processed_face)
        else:
            print("No face detected in the current frame.")
    
    # Compute heart rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    peaks, _ = find_peaks(rppg_signal, distance=fps/2)
    peak_intervals = np.diff(peaks) / fps
    heart_rate = 60.0 / np.mean(peak_intervals)
    print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
    
    plt.close()  # Close the figure when done

cap.release()