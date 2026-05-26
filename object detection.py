from ultralytics import YOLO
import cv2

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    # Read Frame
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO Detection
    results = model(frame)

    # Draw Results
    annotated_frame = results[0].plot()

    # Title
    cv2.putText(
        annotated_frame,
        "AI Object Detection",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        3
    )

    # Show Window
    cv2.imshow(
        "YOLO Object Detection",
        annotated_frame
    )

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()