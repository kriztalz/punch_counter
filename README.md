# Punch Counter

A lightweight web application that uses TensorFlow.js and MoveNet to accurately count forward punches detected through your webcam.

## Features

- Real-time forward punch detection using webcam feed
- Pose estimation using TensorFlow.js MoveNet model
- Clean white skeleton overlay to show detected body positions
- Punch counter with visual feedback effects
- Simple, minimalist UI
- Detection of only proper forward punches (toward the camera)
- Smart detection that requires arm retraction between punches

## How It Works

1. The application uses the MoveNet pose detection model from TensorFlow.js to identify body keypoints
2. It tracks the movement of your wrists and calculates their velocity and direction
3. Multiple algorithms work together to detect true forward punches:
   - Analyzing the apparent change in arm length (forward movement)
   - Tracking wrist-to-elbow movement ratios
   - Monitoring alignment of wrist with shoulders
   - Measuring relative position changes
4. The system requires your arm to return to a retracted position between punches
5. When a valid punch is detected, the counter increments with visual feedback

## Usage

1. Open `index.html` in a modern web browser
2. Allow camera access when prompted
3. Click the "Start Camera" button to begin
4. Stand back so your upper body is visible in the frame
5. Throw forward punches directly toward the camera
6. The counter will only increment for proper forward punches
7. You must retract your arm between punches for them to count
8. Use the "Reset Counter" button to start counting from zero again
9. Click "Stop Camera" when finished to disable the webcam

## Requirements

- Modern web browser with JavaScript enabled
- Webcam access
- Internet connection (to load TensorFlow.js libraries)

## Privacy

All processing is done locally in your browser. No video data is sent to any servers.

## Technical Implementation

- Uses TensorFlow.js for machine learning capabilities
- Leverages the MoveNet SINGLEPOSE_LIGHTNING model for fast pose detection
- Implements sophisticated multi-method punch detection algorithms
- Uses velocity, position and geometric analysis to identify true forward punches
- Requires arm retraction between punches to prevent double-counting
- Uses HTML5 Canvas for rendering pose visualization
- Written in vanilla JavaScript with no additional dependencies 