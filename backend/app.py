"""
Flask Application - Main Server
Handles web interface, video streaming, and text-to-speech
"""
from flask import Flask, render_template, Response, jsonify, request
import cv2
from gesture_recognition import GestureRecognizer
import pyttsx3
import threading
import os

# Configure paths for frontend files
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Initialize gesture recognizer
gesture_recognizer = GestureRecognizer()

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

# Global variables
camera = None
current_gesture = "No hand detected"
accumulated_text = ""
last_spoken_gesture = None  # Track last spoken gesture to prevent repetition

def get_camera():
    """Initialize camera if not already initialized"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

# Add this global variable at top with others
is_speaking = False


def speak_gesture(text):
    """Speak gesture text safely (no overlapping speech)"""
    global is_speaking

    if is_speaking:
        return  # Don't speak if already speaking

    def speak():
        global is_speaking
        try:
            is_speaking = True
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            is_speaking = False

    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()

def generate_frames():
    """Generator function to yield video frames"""
    global current_gesture, last_spoken_gesture
    
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Recognize gesture
        processed_frame, gesture_text = gesture_recognizer.recognize_gesture(frame)
        print("MODEL OUTPUT:", gesture_text)
        current_gesture = gesture_text
        
        # Speak gesture if it's new (different from last spoken gesture)
        # and not a "no hand" or "unknown" state
        if (gesture_text != "No hand detected" and 
            gesture_text != "Unknown" and 
            gesture_text != last_spoken_gesture and not is_speaking):
            speak_gesture(gesture_text)
            last_spoken_gesture = gesture_text
        
        # Reset if no hand detected
        if gesture_text == "No hand detected":
            last_spoken_gesture = None
        
        # Add text to frame
        cv2.putText(
            processed_frame, 
            f"Gesture: {gesture_text}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_gesture')
def get_gesture():
    """Get current gesture as JSON"""
    return jsonify({'gesture': current_gesture})

@app.route('/add_to_text', methods=['POST'])
def add_to_text():
    """Add current gesture to accumulated text"""
    global accumulated_text, current_gesture
    
    if current_gesture != "No hand detected" and current_gesture != "Unknown":
        # Extract just the word/meaning from gesture text
        text_to_add = current_gesture.split(' - ')[-1] if ' - ' in current_gesture else current_gesture
        
        if accumulated_text:
            accumulated_text += " " + text_to_add
        else:
            accumulated_text = text_to_add
    
    return jsonify({'text': accumulated_text})

@app.route('/get_text')
def get_text():
    """Get accumulated text"""
    return jsonify({'text': accumulated_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    """Clear accumulated text"""
    global accumulated_text
    accumulated_text = ""
    return jsonify({'status': 'success'})

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech using pyttsx3"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Speak text in a background thread
        def speak_text():
            tts_engine.say(text)
            tts_engine.runAndWait()
        
        thread = threading.Thread(target=speak_text)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'success', 'text': text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    print("Starting Assistive Communication System...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
