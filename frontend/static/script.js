// Update current gesture display every 500ms
let gestureUpdateInterval = setInterval(updateGesture, 500);

function updateGesture() {
    fetch('/get_gesture')
        .then(response => response.json())
        .then(data => {
            document.getElementById('current-gesture').textContent = data.gesture;
        })
        .catch(error => console.error('Error fetching gesture:', error));
}

// Add current gesture to text
document.getElementById('add-btn').addEventListener('click', function() {
    fetch('/add_to_text', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('accumulated-text').value = data.text;
    })
    .catch(error => console.error('Error adding gesture:', error));
});

// Add space to text
document.getElementById('space-btn').addEventListener('click', function() {
    let textArea = document.getElementById('accumulated-text');
    let currentText = textArea.value;
    
    if (currentText.length > 0 && !currentText.endsWith(' ')) {
        textArea.value = currentText + ' ';
    }
});

// Clear text
document.getElementById('clear-btn').addEventListener('click', function() {
    if (confirm('Are you sure you want to clear all text?')) {
        fetch('/clear_text', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('accumulated-text').value = '';
        })
        .catch(error => console.error('Error clearing text:', error));
    }
});

// Text to Speech
document.getElementById('speak-btn').addEventListener('click', function() {
    let text = document.getElementById('accumulated-text').value.trim();
    
    if (text === '') {
        alert('Please add some text first!');
        return;
    }
    
    // Show loading state
    let speakBtn = document.getElementById('speak-btn');
    let originalText = speakBtn.textContent;
    speakBtn.textContent = '⏳ Speaking...';
    speakBtn.disabled = true;
    
    fetch('/text_to_speech', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.audio) {
            // Create audio element and play
            let audio = new Audio('data:audio/mp3;base64,' + data.audio);
            audio.play();
            
            // Reset button when audio finishes
            audio.onended = function() {
                speakBtn.textContent = originalText;
                speakBtn.disabled = false;
            };
        } else if (data.error) {
            alert('Error: ' + data.error);
            speakBtn.textContent = originalText;
            speakBtn.disabled = false;
        }
    })
    .catch(error => {
        console.error('Error with text-to-speech:', error);
        alert('Error generating speech. Please try again.');
        speakBtn.textContent = originalText;
        speakBtn.disabled = false;
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter: Add gesture
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        document.getElementById('add-btn').click();
    }
    
    // Ctrl/Cmd + Space: Speak
    if ((event.ctrlKey || event.metaKey) && event.key === ' ') {
        event.preventDefault();
        document.getElementById('speak-btn').click();
    }
});

// Load initial text on page load
window.addEventListener('load', function() {
    fetch('/get_text')
        .then(response => response.json())
        .then(data => {
            document.getElementById('accumulated-text').value = data.text;
        })
        .catch(error => console.error('Error loading text:', error));
});

// Clean up when page is closed
window.addEventListener('beforeunload', function() {
    clearInterval(gestureUpdateInterval);
    fetch('/stop_camera', { method: 'POST' });
});
