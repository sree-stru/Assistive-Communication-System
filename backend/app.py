import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# ✅ Load trained model once
with open("model.p", "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]
label_encoder = model_dict["label_encoder"]

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Preprocess
        img = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten().reshape(1, -1)

        # Predict
        prediction = model.predict(flat)
        label = label_encoder.inverse_transform(prediction)[0]

        # Show prediction on frame
        cv2.putText(frame, f"Prediction: {label}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)