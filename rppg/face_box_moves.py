import cv2
import matplotlib.pyplot as plt

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

video_path = "vid.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    plt.figure(figsize=(12, 6))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_face, face_coords = preprocess_frame(frame)
        
        if processed_face is not None:
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
            plt.pause(0.1)  # Pause to update the plot
        else:
            print("No face detected in the current frame.")
    
    plt.close()  # Close the figure when done

cap.release()