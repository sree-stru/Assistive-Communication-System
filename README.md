# Indian Sign Language (ISL) Landmarks Interpreter

A real-time, high-accuracy Indian Sign Language (ISL) gesture recognition web app powered by hand-joint coordinates and Gemini AI.

## Features
- **🧠 Landmarks-Based AI**: Uses 21 hand-joint coordinates (42 features) for classification.
- **🛡️ 100% Background Proof**: Works in any room, lighting, or skin tone because it ignores pixels.
- **🚀 99.78% Accuracy**: Trained on 41,000+ vectorized samples across 35 classes (A-Z, 1-9).
- **🌐 Web Interface**: Glassmorphism-styled real-time web app powered by Flask.
- **🤖 Gemini AI Refinement**: Noisy gesture sequences are auto-corrected into meaningful sentences.
- **✍️ Text → Sign Language**: Type any sentence; Gemini extracts keywords and animates the ISL hand skeleton letter-by-letter on a canvas.
- **🔊 Robust TTS**: Non-blocking, thread-safe text-to-speech engine.
- **📱 Word Autocomplete**: Predictive text suggestions for faster communication.

---

## Project Structure
```
assistive communication/
├── config.py                   ← Central configuration (reads env vars)
├── .env.example                ← Copy to .env and add your API key
├── src/
│   ├── web_server.py           ← Flask web server & all API endpoints
│   ├── gesture_detector.py     ← MediaPipe Landmarks Extractor
│   ├── gemini_service.py       ← Gemini AI sentence refinement + keyword extraction
│   ├── sign_landmark_data.py   ← Landmark lookup for Text→Sign animation
│   ├── tts.py                  ← Worker-thread TTS Engine
│   ├── autocomplete.py         ← Predictive text engine
│   ├── model.py                ← Landmark-based Dense Architecture
│   ├── train_landmarks.py      ← Training with Augmentation
│   └── extract_landmarks.py    ← Dataset vectorizer (Parallel)
├── templates/
│   └── index.html              ← Main web UI
├── models/                     ← sign_landmark_model.keras & hand_landmarker.task
└── requirements.txt
```

---

## Setup & Usage

### 1. Clone & Install
```bash
git clone https://github.com/Anjana2113/Sign-Language-Predicting-System.git
cd Sign-Language-Predicting-System
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Your Gemini API Key
```bash
# Copy the example file
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux

# Then open .env and paste your key:
# GEMINI_API_KEY=your_gemini_api_key_here
```
> Get a free key at: https://aistudio.google.com/app/apikey

### 3. Run the Web App
```bash
python src/web_server.py
```
Then open your browser at **http://localhost:5000**

---

## How to Use
| Feature | How |
|---|---|
| **Sign → Text** | Show an ISL hand sign to your webcam; click **Append** to add the letter |
| **Refine with AI** | Click **Refine with AI** to auto-correct the detected noise into a sentence |
| **Speak** | Click **Speak Sentence** to hear the text read aloud |
| **Text → Sign** | Scroll down, type a sentence, click **Convert to Sign ✨** |

---

## Technical Details
- **Architecture**: 3-layer Dense Neural Network (ReLU + Dropout)
- **Feature Extraction**: MediaPipe Tasks Vision API (v0.10.0+)
- **Normalization**: Translation-invariant (wrist-centered) and Scale-invariant (Max-Abs scaling)
- **AI Backend**: Google Gemini API (gemini-2.0-flash / gemini-1.5-flash)
