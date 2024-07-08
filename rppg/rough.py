import os
import cv2
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from scipy.signal import find_peaks


# 1. Define Masked Autoencoder Architecture
def build_masked_autoencoder(input_shape):
    # Input layer that takes in the shape of the images
    input_layer = Input(shape=input_shape)
    
    # Flatten the input image to a 1D array
    flattened_input = Flatten()(input_layer)
    
    # Encoder: Reduce dimensions using Dense layers
    encoded = Dense(128, activation='relu')(flattened_input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    # Decoder: Reconstruct the image back to original shape
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(decoded)
    decoded = Reshape(input_shape)(decoded)
    
    # Create the autoencoder model
    autoencoder = Model(input_layer, decoded)
    # Compile the model with Adam optimizer and Mean Squared Error loss
    autoencoder.compile(optimizer=Adam(), loss=MeanSquaredError())
    return autoencoder

# 2. Data Preparation (Face Detection, Normalization, Signal Extraction)
def preprocess_frame(frame):
    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # If no face is detected, return None
    if len(faces) == 0:
        return None
    
    # Get the coordinates of the first detected face
    (x, y, w, h) = faces[0]
    # Extract the face region from the frame
    face = frame[y:y+h, x:x+w]
    # Resize the face to the desired input shape
    face = cv2.resize(face, (64, 64))
    # Normalize the pixel values to [0, 1]
    face = face / 255.0
    return face

def extract_rppg_signal(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess each frame to extract the face region
        face = preprocess_frame(frame)
        if face is not None:
            frames.append(face)
    cap.release()
    return np.array(frames)

# 3. Train Masked Autoencoder
def train_autoencoder(autoencoder, frames, epochs=50, batch_size=16):
    # Train the autoencoder with the face frames
    autoencoder.fit(frames, frames, epochs=epochs, batch_size=batch_size, shuffle=True)
    return autoencoder

# 4. Extract rPPG Signals and Compute Heart Rate
def extract_rppg_from_model(autoencoder, frames):
    # Use the trained autoencoder to predict the reconstructed frames
    decoded_frames = autoencoder.predict(frames)
    # Extract the green channel from each frame
    green_channel = decoded_frames[:,:,:,1]
    # Compute the average green channel value for each frame
    avg_green = np.mean(green_channel, axis=(1, 2))
    return avg_green

def compute_heart_rate(rppg_signal, fps):
    # Find peaks in the rPPG signal to identify heartbeats
    peaks, _ = find_peaks(rppg_signal, distance=fps/2)
    # Compute the time intervals between peaks
    peak_intervals = np.diff(peaks) / fps
    # Calculate the heart rate from the average peak interval
    heart_rate = 60.0 / np.mean(peak_intervals)
    return heart_rate

# Example Usage
if __name__ == "__main__":
    # Path to the input video file
    video_path = r"vid.avi"
    input_shape = (64, 64, 3)
    
    # Build the autoencoder model
    autoencoder = build_masked_autoencoder(input_shape)
    # Extract face frames from the video
    frames = extract_rppg_signal(video_path)
    
    # Check if any frames were extracted
    if frames.size == 0:
        print("No valid frames extracted from the video.")
    else:
        # Train the autoencoder with the extracted frames
        autoencoder = train_autoencoder(autoencoder, frames)
        # Extract the rPPG signal using the trained autoencoder
        rppg_signal = extract_rppg_from_model(autoencoder, frames)
        
        # Define the frames per second (FPS) of the video
        fps = 30  # Adjust based on your video fps
        # Compute the heart rate from the rPPG signal
        heart_rate = compute_heart_rate(rppg_signal, fps)
        print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")

        # Save the result to a file in the specified directory
        output_directory = r"C:/Users/SHUBH SANKALP DAS/Desktop/SEM 4/SOP/subject1"
        output_file_path = os.path.join(output_directory, "heart_rate_output.txt")
        with open(output_file_path, 'w') as file:
            file.write(f"Estimated Heart Rate: {heart_rate:.2f} BPM\n")
            file.write("rPPG Signal:\n")
        
        # Open the output file with the default application
        try:
            os.startfile(output_file_path)  # Windows
        except AttributeError:
            try:
                subprocess.run(['open', output_file_path], check=True)  # macOS
            except (FileNotFoundError, subprocess.CalledProcessError):
                subprocess.run(['xdg-open', output_file_path], check=True)  # Linux
