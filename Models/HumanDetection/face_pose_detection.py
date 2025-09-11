import pickle
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh once (avoid reloading in every function call)
facelib = mp.solutions.face_mesh
face_mesh = facelib.FaceMesh(static_image_mode=True)

# Load the trained model
loaded_model = pickle.load(open('face_vector_model.sav', 'rb'))

def classify_pose(predictions):
    categorized_poses = []

    for pred in predictions:
        y, x, z = pred  # Assuming prediction format is [x, y, z]

        if y > 0.15 and abs(x) < 0.1:
            pose = "Pose Up"

        elif y < -0.1 and abs(x) < 0.15:
            pose = "Pose Down"

        elif x > 0.1 and abs(y) < 0.1:
            pose = "Pose Right"

        elif x < -0.4 and abs(y) < 0.1:
            pose = "Pose Left"

        elif x > 0.1 and y > 0.1:
            pose = "Pose Top Right"

        elif x < -0.2 and y > 0.08:
            pose = "Pose Top Left"

        elif x > 0.1 and y < -0.01:
            pose = "Pose Bottom Right"

        elif x < -0.3 and y < -0.1:
            pose = "Pose Bottom Left"

        else:
            pose = "Neutral"

        categorized_poses.append(pose)

    return categorized_poses


def preprocess(frame):

    # Ensure frame is in uint8 format (fix OpenCV error)
    if frame.dtype == np.float64:
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Convert to RGB (MediaPipe requires RGB input)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with FaceMesh
    results = face_mesh.process(image_rgb)

    # If no faces detected, return None
    if results.multi_face_landmarks is None:
        print('No faces detected')
        return None

    # Get face landmarks
    landmarks = results.multi_face_landmarks[0].landmark
    height, width = image_rgb.shape[:2]

    x_val = np.array([lm.x * width for lm in landmarks])
    y_val = np.array([lm.y * height for lm in landmarks])

    # Centering around the nose landmark (landmark index 1)
    x_val -= np.mean(x_val)
    y_val -= np.mean(y_val)

    # Normalize based on the maximum distance
    x_max, y_max = x_val.max(), y_val.max()

    if x_max != 0:
        x_val /= x_max
    if y_max != 0:
        y_val /= y_max

    return np.concatenate([x_val, y_val])

if __name__ == "__main__":
    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_data = preprocess(frame)

        if processed_data is not None:
            processed_data = processed_data.reshape(1, -1)
            prediction = loaded_model.predict(processed_data)
            print("Prediction:",classify_pose(prediction))

    cap.release()
    cv2.destroyAllWindows()





