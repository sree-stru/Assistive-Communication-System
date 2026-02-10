# 🤝 Assistive Communication System for Speech and Hearing Impaired

A real-time hand gesture recognition system that converts hand gestures to text and speech. Built with Python, MediaPipe, OpenCV, and Flask.

## 📋 Project Overview

This mini project helps speech and hearing impaired individuals communicate through hand gestures. The system:
- Detects hand gestures in real-time using a webcam
- Converts gestures to text
- Converts accumulated text to speech
- Provides a simple web-based interface

## 🗂️ Project Structure

```
Mini project/
│
├── backend/                  # Backend Python files
│   ├── app.py               # Main Flask server (starts the web application)
│   ├── gesture_recognition.py  # Hand gesture detection and recognition logic
│   ├── collect_data.py      # Optional: For training custom gestures
│   ├── requirements.txt     # Python dependencies
│   └── models/              # Trained models will be saved here
│
├── frontend/                 # Frontend files
│   ├── templates/
│   │   └── index.html       # Main web page
│   └── static/
│       ├── style.css        # Styling for the web interface
│       └── script.js        # Frontend JavaScript (handles user interactions)
│
└── README.md                 # This file
```

## 📥 Dataset

The training datasets are stored separately to keep the repository size manageable.

### 1) Digits and Alphabets (dat.zip)
**Download link:** [https://drive.google.com/open?id=1keWr7-X8aR4YMotY2m8SlEHlyruDDdVi](https://drive.google.com/open?id=1keWr7-X8aR4YMotY2m8SlEHlyruDDdVi)

After downloading:
1. Download `data.zip` from the Google Drive link
2. Extract `data.zip`
3. Place the extracted folder inside `backend/dataset/Indian/`

### 2) Images for Phrases
**Download link:** [https://data.mendeley.com/datasets/w7fgy7jvs8/3](https://data.mendeley.com/datasets/w7fgy7jvs8/3)

After downloading:
1. Extract the dataset folder
2. Place it inside `backend/dataset/images for phrases/`
3. Make sure the folder structure matches the expected format (see dataset README for details)

## 📄 What Each File Does

### Core Files:

1. **app.py** - Main application server
   - Runs the Flask web server
   - Handles video streaming from webcam
   - Processes gesture recognition requests
   - Converts text to speech
   - Manages communication between frontend and backend

2. **gesture_recognition.py** - Gesture detection engine
   - Uses MediaPipe to detect hand landmarks
   - Recognizes gestures based on finger positions
   - Currently supports 6 basic gestures (Fist, 1-5 fingers)
   - Can be extended with machine learning models

3. **requirements.txt** - Python packages needed
   - Lists all dependencies with versions
   - Used for easy installation with pip

4. **index.html** - Web interface
   - Shows live camera feed
   - Displays recognized gestures
   - Has buttons to add text, speak, and clear

5. **style.css** - Visual styling
   - Makes the interface look good
   - Responsive design for different screen sizes

6. **script.js** - Frontend logic
   - Updates gesture display in real-time
   - Sends requests to backend
   - Plays generated speech audio

7. **collect_data.py** - (Optional) Custom gesture training
   - Collect your own gesture samples
   - Train a machine learning model
   - For advanced users who want custom gestures

## 🚀 How to Run the Project

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed on your computer.
Check by opening PowerShell and typing:
```powershell
python --version
```

### Step 2: Open Project Folder
Open PowerShell and navigate to the backend folder:
```powershell
cd "d:\Mini project\backend"
```

### Step 3: Install Dependencies
Copy and paste this command to install all required packages:
```powershell
pip install -r requirements.txt
```

Wait for all packages to download and install (may take 2-5 minutes).

### Step 4: Run the Application
Start the server with:
```powershell
python app.py
```

You should see:
```
Starting Assistive Communication System...
Open http://localhost:5000 in your browser
```

### Step 5: Open in Browser
Open your web browser (Chrome, Edge, Firefox) and go to:
```
http://localhost:5000
```

### Step 6: Allow Camera Access
- Your browser will ask for camera permission
- Click "Allow" to let the application use your webcam

## 🎯 How to Use the System

1. **Show Hand Gesture** - Hold your hand in front of the webcam
   - Make sure your hand is clearly visible
   - Good lighting helps with detection

2. **View Recognition** - See the recognized gesture below the video

3. **Add to Text** - Click "➕ Add Gesture" button to add the recognized word to the text area

4. **Build Sentence** - Keep adding gestures to build your message

5. **Speak** - Click "🔊 Speak" button to hear your message out loud

6. **Clear** - Click "🗑️ Clear" button to start over

## 🖐️ Supported Gestures

| Hand Gesture | Meaning |
|--------------|---------|
| Fist (0 fingers) | YES |
| 1 finger up | I |
| 2 fingers up | NEED |
| 3 fingers up | HELP |
| 4 fingers up | WAIT |
| 5 fingers up (open palm) | HELLO |

## 🔧 Troubleshooting

### Camera Not Working
- Make sure no other application is using the camera
- Check camera permissions in Windows settings
- Try restarting the application

### Gestures Not Recognized
- Ensure good lighting
- Keep hand centered in the video
- Try moving hand closer/farther from camera
- Make gestures clear and distinct

### Installation Errors
- Update pip: `python -m pip install --upgrade pip`
- Try installing packages one by one if bulk install fails
- Make sure you have stable internet connection

### Port Already in Use
If you see "Port 5000 is already in use":
```powershell
# Find and kill the process using port 5000
netstat -ano | findstr :5000
taskkill /PID <process_id> /F
```

## 🎓 For Presentation/Demo

1. Open the application
2. Show the live camera feed detecting your hand
3. Demonstrate each gesture (0-5 fingers)
4. Build a simple sentence: "I NEED HELP"
5. Click Speak to convert to speech
6. Explain the technology: MediaPipe, OpenCV, Flask

## 🔮 Future Enhancements (Optional)

- Add more custom gestures using `collect_data.py`
- Train ML model for better accuracy
- Add sign language alphabet support
- Add gesture history/common phrases
- Multi-hand gesture support
- Mobile app version

## 📚 Technologies Used

- **Python 3.8+** - Programming language
- **Flask** - Web framework
- **OpenCV** - Computer vision library
- **MediaPipe** - Hand tracking and detection
- **NumPy** - Numerical computations
- **gTTS** - Google Text-to-Speech
- **scikit-learn** - Machine learning (for custom training)

## 👨‍💻 Development Notes

- The system uses rule-based gesture recognition by default
- Simple and lightweight for mini project requirements
- Can be extended with ML models for more gestures
- Designed for easy understanding and modification

## 📞 Support

If you face any issues:
1. Make sure all dependencies are installed
2. Check that your camera is working
3. Verify Python version is 3.8 or higher
4. Restart the application
5. Check the terminal for error messages

## 📝 License

This is a mini project for educational purposes.

---

**Good luck with your project! 🚀**
