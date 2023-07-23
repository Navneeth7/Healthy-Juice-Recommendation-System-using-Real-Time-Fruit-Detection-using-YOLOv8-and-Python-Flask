import cv2

camera = cv2.VideoCapture(1)  # Change the index as needed (0, 1, 2, etc.)

if not camera.isOpened():
    print("Camera not opened. Check camera connection and index.")
else:
    success, frame = camera.read()
    if success:
        print("Camera is accessible.")
    else:
        print("Failed to read a frame from the camera.")

camera.release()
