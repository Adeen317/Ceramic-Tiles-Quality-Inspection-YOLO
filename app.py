import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import serial
import uuid   # Unique identifier
import os
import time

# Convert resources.qrc file to resources.py for labelImg
os.system('cd labelImg && pyrcc5 -o libs/resources.py resources.qrc')

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/best.pt', force_reload=True)

# Set up serial communication with Arduino
ser = serial.Serial('COM4', 9600, timeout=1)  # Replace 'COM4' with your Arduino port
time.sleep(2)  # Wait for the serial connection to initialize

IMAGES_PATH = os.path.join('data', 'defected_images')
IMAGES_PATH1 = os.path.join('data', 'defected_images_converted')

# Ensure the directories exist
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH1, exist_ok=True)

# Open video capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Grayscale
    gray_frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, threshold1 = cv2.threshold(gray_frame1, 90, 255, cv2.THRESH_BINARY)

    # Invert Image
    invert1 = cv2.bitwise_not(threshold1)

    # Assuming you have a model defined and loaded
    results1 = model(invert1)

    h = np.hstack((frame, results1.render()[0]))  # Ensure correct syntax for np.hstack
    cv2.imshow("end", h)

    if ser.in_waiting > 0:  # Check if there's a message from Arduino
        try:
            msg = ser.readline().decode('utf-8', errors='ignore').strip()  # Read the message
            print(f"Received message: {msg}")  # Debug print
        except UnicodeDecodeError as e:
            print(f"Error decoding message: {e}")
            continue  # Skip the rest of the loop and try again

        if "tile is detected" in msg:  # Check if the message contains the keyword
            imgname = os.path.join(IMAGES_PATH, f"captured_image_{time.time()}.png")
            cv2.imwrite(imgname, frame)  # Save the captured image
            print(f"Image saved as {imgname}")
            ser.write(b'image captured\n')  # Send acknowledgment back to Arduino

            # Read the saved image
            image = cv2.imread(imgname)

            # Grayscale
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Thresholding
            _, threshold = cv2.threshold(gray_frame, 88, 255, cv2.THRESH_BINARY)

            # Invert Image
            invert = cv2.bitwise_not(threshold)

            # Make detections 
            results = model(invert)
            
            cv2.imshow('YOLO', np.squeeze(results.render()))

            imgname1 = os.path.join(IMAGES_PATH1, f"captured_image_{time.time()}.png")
            cv2.imwrite(imgname1, np.squeeze(results.render()))  # Save the captured image
            print(f"Image saved as {imgname1}")

            # Get the number of detected cracks
            num_cracks = len(results.xyxy[0])

            # Print message if cracks are detected
            if num_cracks > 0:
                print("Defect detected")
                ser.write(b'tile is defected\n')  # Send defect signal to Arduino
            else:
                print("No defect detected")

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
