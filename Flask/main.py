from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/last.pt', force_reload=True)

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Grayscale
    gray_frame1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, threshold1 = cv2.threshold(gray_frame1, 90, 255, cv2.THRESH_BINARY)

    # Invert Image
    invert1 = cv2.bitwise_not(threshold1)
    contours, _ = cv2.findContours(invert1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if 5 < area < 1000:
            cv2.rectangle(invert1, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(invert1, 'Spot', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # Model inference
    results1 = model(invert1)

    # Combine original image and detection results
    h = np.hstack((img, results1.render()[0]))  # Ensure correct syntax for np.hstack

    # Save the processed image
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, h)

    return processed_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + '_' + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            processed_image_path = process_image(file_path)
            return render_template('index.html', uploaded_image=file_path, processed_image=processed_image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
