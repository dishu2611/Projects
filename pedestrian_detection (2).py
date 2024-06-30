import cv2
import pygame
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from imutils.object_detection import non_max_suppression

# Initialize pygame mixer
pygame.mixer.init()

# Load the beep sound
beep_sound = pygame.mixer.Sound("beep-warning-6387.mp3")  # replace with your beep sound file

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize Tkinter window
root = tk.Tk()
root.title('Pedestrian Detection')

# Create a label widget for video frame and popup message
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

# Function to detect pedestrians and update button messages
def detect_pedestrians():
    # Open the webcam
    cap = cv2.VideoCapture(1)  # 0 for the first webcam, 1 for the second, and so on

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error opening webcam")

    while cap.isOpened():
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to improve processing speed and detection accuracy
        frame = cv2.resize(frame, (640, 480))

        # Detect pedestrians in the frame
        boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        if len(boxes) > 0:
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            for (startX, startY, endX, endY) in pick:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                # Print alert and play beep sound when a pedestrian is detected
                print("Alert: Pedestrian detected!")
                beep_sound.play()

                # Display "Stop" button in red
                cv2.rectangle(frame, (10, 10), (100, 50), (0, 0, 255), -1)
                cv2.putText(frame, "Stop", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display "Go" button in green if no pedestrians are detected
        if len(boxes) == 0:
            cv2.rectangle(frame, (10, 10), (100, 50), (0, 255, 0), -1)
            cv2.putText(frame, "Go", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert the frame to RGB format for displaying in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the video frame in the Tkinter label widget
        video_label.img_tk = img_tk
        video_label.config(image=img_tk)

        # Wait for 1 millisecond to refresh the frame
        root.update_idletasks()
        root.update()

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

# Button to start pedestrian detection
start_button = tk.Button(root, text="Start Detection", command=detect_pedestrians)
start_button.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()
