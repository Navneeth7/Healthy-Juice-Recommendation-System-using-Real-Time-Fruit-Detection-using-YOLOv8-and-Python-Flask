from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model=YOLO("C:/Users/navne/Desktop/fruit_train/runs/detect/train13/weights/best.pt")
model.predict(source="0", show=True, conf=0.5)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

cv2.destroyAllWindows()
