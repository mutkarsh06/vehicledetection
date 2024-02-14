import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load video capture
cap = cv2.VideoCapture('/content/drive/MyDrive/Projectmini/yt.mp4')

# Initialize background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize K-Nearest Neighbors classifier for color detection
color_classifier = KNeighborsClassifier(n_neighbors=4)  # Increase the number of neighbors for better classification

# Load pre-trained color dataset
# For demonstration purposes, let's create a synthetic dataset
# Replace this with your actual dataset loading code
colors = ['red', 'green', 'blue', 'white', 'black']
color_values = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]

# Train the color classifier
color_classifier.fit(color_values, colors)

# Initialize variables for speed detection
previous_frame = None
previous_bbox = None
speeds = []

# Define font and text parameters for speed overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
line_thickness = 2
speed_text_position = (50, 50)

# Define output video parameters
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/content/drive/MyDrive/Projectmini/Modelcheck_output.avi', fourcc, output_fps, (output_width, output_height))

# Initialize bounding box coordinates
x, y, w, h = 0, 0, 0, 0

# Loop through video frames
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fg_mask = background_subtractor.apply(gray_frame)

    # Morphological operations to remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours of detected objects
    contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset bounding box coordinates
    x, y, w, h = 0, 0, 0, 0

    # Loop through detected contours
    for contour in contours:
        # Filter contours based on area
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        vehicle_image = frame[y:y+h, x:x+w]

        vehicle_image = cv2.resize(vehicle_image, (100, 100))

        mean_color = np.mean(vehicle_image, axis=(0, 1))
        color_name = color_classifier.predict([mean_color])[0]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Color: {color_name}', (x, y-10), font, font_scale, (0, 255, 0), line_thickness)

        if previous_bbox is not None:
            prev_x, prev_y, prev_w, prev_h = previous_bbox
            curr_x, curr_y, curr_w, curr_h = x, y, w, h
            prev_center = np.array([(prev_x + prev_w) / 2, (prev_y + prev_h) / 2])
            curr_center = np.array([(curr_x + curr_w) / 2, (curr_y + curr_h) / 2])
            speed = np.linalg.norm(prev_center - curr_center)
            speeds.append(speed)

    if speeds:
        average_speed = int(np.mean(speeds))
        cv2.putText(frame, f'Average Speed: {average_speed} per frame(m/s)', speed_text_position, font, font_scale, font_color, line_thickness)

    out.write(frame)

    previous_frame = gray_frame.copy()
    previous_bbox = (x, y, w, h)

cap.release()
out.release()
cv2.destroyAllWindows()
