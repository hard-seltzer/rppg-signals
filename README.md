# rPPG Signal Extraction and Heart Rate Estimation

This project contains various Python scripts for extracting remote photoplethysmography (rPPG) signals from video files and estimating heart rates. Below is an overview of the files in the rppg folder and their functionalities.

## Files and Their Functions

### 1. main.py
This is the main script that implements rPPG signal extraction using a masked autoencoder.

Steps:
1. Opens the video file.
2. Initializes a plot for displaying results.
3. Builds the masked autoencoder model.
4. Processes frames from the video:
   - Extracts face region
   - Applies the autoencoder to extract rPPG signal
   - Updates the plot with current frame, processed face, and rPPG signal
5. Trains the autoencoder periodically with buffered frames.
6. Computes heart rate from the extracted rPPG signal.
7. Prints the estimated heart rate.

### 2. rough.py
This script contains a more detailed implementation of the rPPG signal extraction process, including data preparation and model training.

Steps:
1. Defines the masked autoencoder architecture.
2. Implements data preparation functions (face detection, normalization).
3. Defines functions for training the autoencoder and extracting rPPG signals.
4. Processes a video file to extract frames and train the model.
5. Computes and saves the estimated heart rate.

### 3. without_autoencoder.py
This script demonstrates rPPG signal extraction without using an autoencoder.

Steps:
1. Opens the video file.
2. Initializes a plot for displaying results.
3. Processes frames from the video:
   - Detects and extracts face region
   - Computes average green channel value as rPPG signal
4. Updates the plot with current frame, processed face, and rPPG signal.
5. Computes heart rate from the extracted rPPG signal.
6. Prints the estimated heart rate.

### 4. face_box_moves.py
This script focuses on visualizing face detection across video frames.

Steps:
1. Opens the video file.
2. Processes each frame:
   - Detects face and draws bounding box
   - Displays the frame with face detection and processed face
3. Updates the plot for each frame.

### 5. preprocess.py
This script demonstrates the preprocessing step for a single frame of video.

Steps:
1. Opens the video file.
2. Reads the first frame.
3. Preprocesses the frame:
   - Detects face
   - Extracts and resizes face region
   - Normalizes pixel values
4. Displays the original frame with face detection and the processed face.

### 6. frame_rate.py
This script focuses on measuring and displaying the frame processing rate.

Steps:
1. Opens the video file.
2. Processes frames in a loop:
   - Detects and extracts face region
   - Displays the frame with face detection and processed face
3. Calculates and prints the frames per second (FPS) every second.


## Requirements

- OpenCV
- NumPy
- Matplotlib
- TensorFlow (for main.py using the autoencoder)
- SciPy

## License

This project is licensed under the MIT License. See the LICENSE file for details.