import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('cheating_object_model.pt')

# Start capturing from webcam
cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()

    if not success:
        break

    # image_path = 'objects.jfif'
    # image = cv2.imread(image_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLOv8

    # Perform YOLO prediction on the frame
    results = model.predict(frame, imgsz=640, conf=0.3)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display annotated frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
