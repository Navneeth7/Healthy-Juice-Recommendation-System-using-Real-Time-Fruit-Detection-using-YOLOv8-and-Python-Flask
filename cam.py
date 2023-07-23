import cv2
from ultralytics import YOLO

# Set the index or address of the external camera as the source
source = "0"  # Replace with the appropriate index or address of your external camera

# Initialize the YOLO model
model = YOLO("C:/Users/navne/Desktop/fruit_train/runs/detect/train13/weights/best.pt")

# Start object detection on the external camera
model.predict(source=source, show=True, conf=0.5)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressedq
        break

cv2.destroyAllWindows()
