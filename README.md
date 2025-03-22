# Punch Counter

A simple web application that uses TensorFlow.js and PoseNet to count punches detected through your webcam.

## Features

- Real-time punch detection using webcam feed
- Pose estimation using TensorFlow.js MoveNet model
- Visual skeleton overlay to show detected body positions
- Punch counter with visual feedback
- Simple, clean UI

## How It Works

1. The application uses the MoveNet pose detection model from TensorFlow.js to identify body keypoints
2. It tracks the movement of your wrists and calculates their velocity
3. When a rapid movement is detected with sufficient arm extension, it registers as a punch
4. A counter displays the total number of punches detected

## Usage

1. Open `index.html` in a modern web browser
2. Allow camera access when prompted
3. Click the "Start Camera" button to begin
4. Stand back so your upper body is visible in the frame
5. Throw punches with either arm
6. The counter will increment for each detected punch
7. Use the "Reset Counter" button to start counting from zero again

## Requirements

- Modern web browser with JavaScript enabled
- Webcam access
- Internet connection (to load TensorFlow.js libraries)

## Privacy

All processing is done locally in your browser. No data is sent to any servers.

## Technical Implementation

- Uses TensorFlow.js for machine learning capabilities
- Leverages the MoveNet SINGLEPOSE_LIGHTNING model for fast pose detection
- Implements velocity-based motion tracking to detect punch movements
- Uses HTML5 Canvas for rendering pose visualization
- Written in vanilla JavaScript with no additional dependencies 