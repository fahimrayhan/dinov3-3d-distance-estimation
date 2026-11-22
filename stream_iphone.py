import cv2
from flask import Flask, Response

app = Flask(__name__)

# Probe indices 0..3 using AVFoundation to find your iPhone Continuity Camera
def open_iphone_camera():
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print(f"Opened camera index {idx} | Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return cap
            cap.release()
    raise RuntimeError("No camera found. Check macOS Privacy & Security -> Camera for VS Code/Terminal.")

cap = open_iphone_camera()

def generate_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # Encode as JPEG for low-latency streaming over SSH
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/frame.jpg")
def single_frame():
    ok, frame = cap.read()
    if not ok:
        return "Camera read failed", 503
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpg.tobytes(), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, threaded=True)