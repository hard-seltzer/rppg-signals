import cv2
import matplotlib.pyplot as plt
import time

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

video_path = "vid.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    plt.figure(figsize=(12, 6))
    face_coords = None
    frame_count = 0
    start_time = time.time()
    
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
            
            plt.clf()  # Clear the current figure
            
            plt.subplot(121)
            plt.imshow(frame_rgb)
            plt.title("Frame with Face Detection")
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(processed_face)
            plt.title("Processed Face")
            plt.axis('off')
            
            plt.tight_layout()
            plt.pause(0.01)   # Pause to update the plot
        else:
            print("No face detected in the current frame.")
        
        # Calculate and print FPS every second
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            print(f"Current FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()
    
    plt.close()  # Close the figure when done

cap.release()