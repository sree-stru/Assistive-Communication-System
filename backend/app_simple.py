"""
Flask Application - Assistive Communication System
Simple hand gesture recognition using OpenCV
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pyttsx3
import threading
import os

# Configure paths for frontend files
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

# Global variables
camera = None
current_gesture = "No hand detected"
accumulated_text = ""

# Simple gesture mapping
GESTURE_MAP = {
    0: "NO",
    1: "YES",
    2: "THANK YOU",
    5: "HELLO"
}


def detect_hand_simple(frame):
    """Simple hand detection using HSV color space"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def count_fingers(mask):
    """Count extended fingers from hand mask"""
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return -1
        
        # Get largest contour (hand)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        if area < 5000:  # Too small to be a hand
            return -1
        
        # Get convex hull
        hull = cv2.convexHull(cnt)
        
        # Count defects to estimate fingers
        defects = cv2.convexityDefects(cnt, hull)
        
        if defects is None:
            return len(hull)
        
        # Count fingers based on defects
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 10000:  # Significant depth for finger separation
                finger_count += 1
        
        return min(finger_count, 5)
        
    except Exception as e:
        print(f"Error counting fingers: {e}")
        return -1


def recognize_gesture_from_mask(mask):
    """Recognize gesture based on hand mask"""
    finger_count = count_fingers(mask)
    
    if finger_count == -1:
        return "No hand detected"
    
    return GESTURE_MAP.get(finger_count, f"Gesture_{finger_count}")


def get_camera():
    """Initialize camera"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def generate_frames():
    """Generate video frames with gesture detection"""
    global current_gesture
    
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand
        mask = detect_hand_simple(frame)
        
        # Recognize gesture
        current_gesture = recognize_gesture_from_mask(mask)
        
        # Display gesture on frame
        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_gesture')
def get_gesture():
    """Get current gesture"""
    return jsonify({'gesture': current_gesture})


@app.route('/add_to_text', methods=['POST'])
def add_to_text():
    """Add gesture to text"""
    global accumulated_text, current_gesture
    
    if current_gesture not in ["No hand detected", "Unknown"]:
        if accumulated_text:
            accumulated_text += " " + current_gesture
        else:
            accumulated_text = current_gesture
    
    return jsonify({'text': accumulated_text})


@app.route('/get_text')
def get_text():
    """Get accumulated text"""
    return jsonify({'text': accumulated_text})


@app.route('/clear_text', methods=['POST'])
def clear_text():
    """Clear text"""
    global accumulated_text
    accumulated_text = ""
    return jsonify({'status': 'success'})


@app.route('/speak', methods=['POST'])
def speak():
    """Convert text to speech"""
    global accumulated_text
    
    if not accumulated_text:
        return jsonify({'status': 'error', 'message': 'No text'})
    
    def speak_text():
        try:
            tts_engine.say(accumulated_text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    thread = threading.Thread(target=speak_text)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success'})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera"""
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    print("=" * 60)
    print("Assistive Communication System - Starting")
    print("=" * 60)
    print("\nServer running on: http://localhost:5000")
    print("\nGesture Mapping:")
    print("  0 fingers (Fist) = NO")
    print("  1 finger = YES")
    print("  2 fingers = THANK YOU")
    print("  5 fingers (Open) = HELLO")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
