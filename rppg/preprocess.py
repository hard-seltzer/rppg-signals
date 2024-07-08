import cv2
import matplotlib.pyplot as plt

def preprocess_frame(frame):
    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
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
    return face, (x, y, w, h)

video_path = "vid.avi"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Read the first frame
    ret, frame = cap.read()
    if ret:
        # Preprocess the frame
        processed_face, face_coords = preprocess_frame(frame)
        
        if processed_face is not None:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Draw rectangle around the detected face
            x, y, w, h = face_coords
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the frame using Matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(frame_rgb)
            ax1.set_title("First Frame with Face Detection")
            ax1.axis('off')
            
            ax2.imshow(processed_face)
            ax2.set_title("Processed Face")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No face detected in the first frame.")
    else:
        print("Error: Could not read the first frame.")

# Release the video capture object
cap.release()