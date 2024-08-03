import cv2
import numpy as np
import os

def preprocess_frame(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, None
    
    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    return face, (x, y, w, h)

def extract_green_channel_average(face):
    green_channel = face[:,:,1]
    return np.mean(green_channel)

video_path = "vid.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    os.makedirs('data', exist_ok=True)
    
    # Load rPPG signal data
    rppg_signal = np.loadtxt('rppg_signal.txt')
    
    # Calculate threshold based on the rPPG signal
    threshold = np.mean(rppg_signal)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_face, face_coords = preprocess_frame(frame)
        
        if processed_face is not None:
            avg_green = extract_green_channel_average(processed_face)
            
            # Use the corresponding rPPG value for this frame
            if frame_count < len(rppg_signal):
                rppg_value = rppg_signal[frame_count]
                label = "High" if rppg_value > threshold else "Low"
                color = (0, 255, 0) if label == "High" else (0, 0, 255)
            else:
                label = "Unknown"
                color = (255, 0, 0)  # Blue for unknown
            
            if face_coords is not None:
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f'Green Channel: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            output_path = os.path.join('data', f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(output_path, frame)
            frame_count += 1
        else:
            print(f"No face detected in frame {frame_count}")
    
    print(f"Processed {frame_count} frames. Annotated images saved in 'data' folder.")

cap.release()