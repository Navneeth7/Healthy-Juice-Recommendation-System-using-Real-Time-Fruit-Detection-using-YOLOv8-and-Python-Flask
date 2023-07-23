from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

# Create a Flask application
app = Flask(__name__)

# OpenCV camera capture
camera = None  # We'll initialize the camera later
model = YOLO("C:/Users/navne/Desktop/fruit_train/runs/detect/train13/weights/best.pt")

# Global variable to keep track of video feed state
video_feed_running = False

def start_video_feed():
    global video_feed_running, camera
    if not video_feed_running:
        # Initialize the camera when starting the feed
        camera = cv2.VideoCapture(1)  # Change 0 to the appropriate camera index if needed
        video_feed_running = True

def stop_video_feed():
    global video_feed_running, camera
    if video_feed_running:
        # Release the camera when stopping the feed
        camera.release()
        video_feed_running = False

def generate_frames():
    start_video_feed()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run object detection on the frame
            results = model.predict(frame)
            
            # print(results)
            # detected_fruits = [names for names, _, _ in results if names in ["apple", "banana", "orange"]]
            # detected_fruits_data = ','.join(detected_fruits)
            
            # f1=results[0].plot()
            # Encode the frame in base64 to embed it in the HTML
            # _, buffer = cv2.imencode('.jpg', frame)
            # frame_base64 = base64.b64encode(buffer).decode()

            # Yield the frame and detected fruits data to the webpage
            ret,buffer=cv2.imencode('.jpg',frame)
            f2=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + f2 + b'\r\n')

    camera.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_feed')
def start_feed():
    start_video_feed()
    return "Video feed started."

@app.route('/stop_feed')
def stop_feed():
    stop_video_feed()
    return "Video feed stopped."

if __name__ == '__main__':
    app.run(debug=True)







