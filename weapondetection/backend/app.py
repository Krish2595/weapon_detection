from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import os, uuid

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

model = YOLO("best4.pt")
model.fuse()


def draw_boxes(frame, results):
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id].lower()

        # ❌ Skip grenade detection
        if "grenade" in label:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = (0,255,0)
        if "gun" in label:
            color = (0,0,255)
        elif "knife" in label:
            color = (0,255,255)

        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,255),2)
    return frame


@app.route("/")
def index():
    return render_template("index.html")



@socketio.on("frame")
def handle_frame(data):
    img = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (640,480))

    results = model(frame, conf=0.5, imgsz=320)
    frame = draw_boxes(frame, results)

    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    encoded = base64.b64encode(buffer).decode()

    emit("response", encoded)


@app.route("/detect-image", methods=["POST"])
def detect_image():
    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.5)
    frame = draw_boxes(frame, results)

    _, buffer = cv2.imencode(".jpg", frame)
    encoded = base64.b64encode(buffer).decode()

    return jsonify({"image": encoded})



@app.route("/detect-video", methods=["POST"])
def detect_video():
    file = request.files["video"]

    input_path = "temp_input.mp4"
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    filename = f"output_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("static/videos", filename)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps if fps > 0 else 20,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, imgsz=320)
        frame = draw_boxes(frame, results)

        out.write(frame)

    cap.release()
    out.release()

    return jsonify({
        "video_url": "/static/videos/" + filename
    })



if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000)

import webbrowser
import threading

def open_browser():
    url = "http://127.0.0.1:5000"

    # Try opening Chrome directly
    chrome_paths = [
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"
    ]

    for path in chrome_paths:
        if os.path.exists(path):
            os.system(f'"{path}" {url}')
            return

    # Fallback → default browser
    import webbrowser
    webbrowser.open(url)


if __name__ == "__main__":
    threading.Timer(2, open_browser).start()
    socketio.run(app, host="127.0.0.1", port=5000)