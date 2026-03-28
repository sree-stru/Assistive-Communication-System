# Indian Sign Language (ISL) Landmarks Interpreter

A real-time, high-accuracy Indian Sign Language (ISL) gesture recognition system powered by hand-joint coordinates.

## Features
- **🧠 Landmarks-Based AI**: Uses 21 hand-joint coordinates (42 features) for classification.
- **🛡️ 100% Background Proof**: Works in any room, lighting, or skin tone because it ignores pixels.
- **🚀 99.78% Accuracy**: Trained on 41,000+ vectorized samples across 35 classes (A-Z, 1-9).
- **📱 Word Autocomplete**: Predictive text suggestions for faster communication.
- **🖱️ Interactive GUI**: On-screen buttons for Add, Space, Speak, Delete, and Clear.
- **🔊 Robust TTS**: Non-blocking, thread-safe text-to-speech engine.

---

## Project Structure
```
assistive communication/
├── config.py               ← Central configuration
├── src/
│   ├── app.py              ← Main Interactive Application
│   ├── gesture_detector.py ← MediaPipe Landmarks Extractor
│   ├── model.py            ← Landmark-based Dense Architecture
│   ├── train_landmarks.py  ← Training with Augmentation
│   ├── extract_landmarks.py← Dataset vectorizer (Parallel)
│   ├── tts.py              ← Worker-thread TTS Engine
│   └── autocomplete.py     ← Predictive text engine
├── models/                 ← sign_landmark_model.keras & hand_landmarker.task
└── requirements.txt        ← tensorflow, mediapipe, opencv-python, pyttsx3
```

---

## Setup & Usage

### 1. Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the App
```bash
python src/app.py
```

### 🖱️ GUI Controls
- **On-Screen Buttons**: Click **ADD**, **SPACE**, **SPEAK**, **DELETE**, or **CLEAR** directly on the webcam window.
- **Suggestions**: Click the **GREEN BOXES** to autocomplete your current word.
- **Keyboard**: 
  - `M`: Toggle Mirroring
  - `Q`: Quit

---

## Technical Details
- **Architecture**: 3-layer Dense Neural Network (ReLU + Dropout)
- **Feature Extraction**: MediaPipe Tasks Vision API (v0.10.0+)
- **Normalization**: Translation-invariant (wrist-centered) and Scale-invariant (Max-Abs scaling).
