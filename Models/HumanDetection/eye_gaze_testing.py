import os
import numpy as np
import cv2
import imutils
import argparse
import tensorflow as tf
import dlib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


def classify_gaze_direction(predictions, threshold=0.005):
    directions = []

    for pred in predictions:
        x, y = pred[0], pred[1]  # Extract x and y coordinates

        if x < -threshold and y > threshold:
            direction = "Looking Top Left"
        elif x > threshold and y > threshold:
            direction = "Looking Top Right"
        elif x < -threshold and y < -threshold:
            direction = "Looking Bottom Left"
        elif x > threshold and y < -threshold:
            direction = "Looking Bottom Right"
        elif x < -threshold:
            direction = "Looking Left"
        elif x > threshold:
            direction = "Looking Right"
        elif y > threshold:
            direction = "Looking Up"
        elif y < -threshold:
            direction = "Looking Down"
        else:
            direction = "Looking Center"

        directions.append(direction)

    return directions

def preprocess_image(img_instance):
    img_instance = cv2.resize(img_instance, (64, 64), interpolation=cv2.INTER_AREA)
    if len(img_instance.shape) == 2:  # Convert grayscale to 3-channel if necessary
        img_instance = cv2.cvtColor(img_instance, cv2.COLOR_GRAY2BGR)
    image = img_instance.astype('float32') / 255.0
    return tf.expand_dims(image, axis=0)

cap = cv2.VideoCapture(1)
predictorPath = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

while True:
    ret,frame = cap.read()
    if not ret:
        break

    # getting the faceattribute vector from dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Define eyebrow top points
        # Get eyebrow top points
        left_brow_y = min(landmarks.part(i).y for i in range(17, 22))
        right_brow_y = min(landmarks.part(i).y for i in range(22, 27))

        # Get lowest eyelid point (bottom eyelid)
        left_lower_eyelid_y = max(landmarks.part(41).y, landmarks.part(40).y) + 15  # Expand downward
        right_lower_eyelid_y = max(landmarks.part(46).y, landmarks.part(47).y) + 15  # Expand downward

        # Define eye bounding box (expanded to include eyebrows, eye corners, and bottom eyelids)
        left_x1 = landmarks.part(36).x - 15  # Expand left eye corner
        left_y1 = left_brow_y  # Use eyebrow top as upper bound
        left_x2 = landmarks.part(39).x + 15  # Expand right eye corner
        left_y2 = left_lower_eyelid_y  # Include bottom eyelid

        right_x1 = landmarks.part(42).x - 15  # Expand left eye corner
        right_y1 = right_brow_y  # Use eyebrow top as upper bound
        right_x2 = landmarks.part(45).x + 15  # Expand right eye corner
        right_y2 = right_lower_eyelid_y  # Include bottom eyelid

        # Crop eye + eyebrow + eye corners + bottom eyelids region
        left_eye_crop = frame[left_y1:left_y2, left_x1:left_x2]
        right_eye_crop = frame[right_y1:right_y2, right_x1:right_x2]

        # Resize the cropped regions for better display
        if left_eye_crop.shape[0] > 0 and left_eye_crop.shape[1] > 0:
            left_eye_crop = cv2.resize(left_eye_crop, (150, 100))
        if right_eye_crop.shape[0] > 0 and right_eye_crop.shape[1] > 0:
            right_eye_crop = cv2.resize(right_eye_crop, (150, 100))

        # Display cropped eye regions
        if left_eye_crop.shape[0] > 0 and left_eye_crop.shape[1] > 0:
            cv2.imshow("Left Eye + Eyebrow + Corners + Bottom Eyelid", left_eye_crop)
        if right_eye_crop.shape[0] > 0 and right_eye_crop.shape[1] > 0:
            cv2.imshow("Right Eye + Eyebrow + Corners + Bottom Eyelid", right_eye_crop)

        resized_eye = cv2.resize(right_eye_crop, (64, 64), interpolation=cv2.INTER_AREA)
        # Load model
        model = load_model("gaze_estimation_model.h5", custom_objects={"mse": MeanSquaredError()})

        # Predict using preprocessed image
        prediction = model.predict(preprocess_image(resized_eye))
        print(prediction)

        # Draw rectangles around the expanded eye regions
        cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), 1)
        cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), 1)

    # Show the original frame with eye detection
    cv2.imshow("Real-Time Eye + Eyebrow + Corners + Bottom Eyelid Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()


