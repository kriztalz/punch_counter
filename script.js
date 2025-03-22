// DOM elements
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const ctx = canvasElement.getContext('2d');
const statusElement = document.getElementById('status');
const counterElement = document.getElementById('counter');
const startButton = document.getElementById('startBtn');
const stopButton = document.getElementById('stopBtn');
const resetButton = document.getElementById('resetBtn');

// Variables for pose detection
let detector;
let model;
let camera;
let rafId;
let punchCount = 0;
let isRunning = false;

// Variables for punch detection
let lastPoseData = null;
let punchState = 'ready'; // 'ready', 'moving', 'cooldown'
let cooldownFrames = 0;
const COOLDOWN_PERIOD = 12; // frames to wait before detecting another punch (reduced from 15)
const VELOCITY_THRESHOLD = 35; // minimum velocity to trigger a punch (reduced from 40)
const EXTENSION_THRESHOLD = 0.12; // how extended the arm needs to be to count as a punch (reduced from 0.15)
const CONFIDENCE_THRESHOLD = 0.4; // minimum confidence score for wrist detection (reduced from 0.6)
const MIN_FRAMES_IN_FRAME = 4; // minimum number of frames wrist must be in frame (reduced from 8)
const FORWARD_MOTION_RATIO = 1.05; // ratio of wrist movement to elbow movement for forward punch (reduced from 1.1)

// Variables to track wrist frames in view
let leftWristFramesInFrame = 0;
let rightWristFramesInFrame = 0;

// New variables to track arm retraction between punches
let leftArmRetracted = true;
let rightArmRetracted = true;
let ARM_RETRACTION_THRESHOLD = 0.08; // threshold for considering arm retracted

// Velocity components analysis for punch detection
const MIN_FORWARD_COMPONENT = 0.3; // minimum percentage of velocity in the forward direction

// For debugging
let debugInfo = {
    leftPunchDirection: null,
    rightPunchDirection: null,
    lastCheck: Date.now()
};

// Initialize the application
async function init() {
    try {
        // Switch to MoveNet which is more reliable
        model = poseDetection.SupportedModels.MoveNet;
        const modelConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
            enableSmoothing: true,
            minPoseScore: 0.2
        };
        
        detector = await poseDetection.createDetector(model, modelConfig);
        statusElement.textContent = 'MoveNet model loaded successfully';
        
        // Enable start button after model is loaded
        startButton.disabled = false;
    } catch (error) {
        console.error('Failed to load the model:', error);
        statusElement.textContent = 'Error loading model: ' + error.message;
    }
}

// Start the webcam
async function startCamera() {
    try {
        const constraints = {
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            }
        };
        
        // Get webcam access
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        
        // Wait for video to be loaded
        return new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                
                // Wait a bit to ensure video is properly playing
                setTimeout(() => {
                    resolve();
                }, 500);
            };
        });
    } catch (error) {
        console.error('Error accessing webcam:', error);
        statusElement.textContent = 'Error accessing webcam: ' + error.message;
        throw error;
    }
}

// Detect poses from webcam feed
async function detectPoses() {
    // Make sure video is ready
    if (!detector || !videoElement.readyState || videoElement.paused || videoElement.videoWidth === 0) {
        console.log('Video not ready yet:', 
            'readyState:', videoElement.readyState, 
            'paused:', videoElement.paused, 
            'width:', videoElement.videoWidth);
        rafId = requestAnimationFrame(detectPoses);
        return;
    }
    
    try {
        // Detect poses
        const poses = await detector.estimatePoses(videoElement);
        console.log('Detected poses:', poses.length > 0 ? 'Yes' : 'No', 
            poses.length > 0 ? `Score: ${poses[0].score.toFixed(2)}` : '');
        
        // Process the detected poses
        processPoses(poses);
        
        // Draw everything
        drawResults(poses);
    } catch (error) {
        console.error('Error during pose detection:', error);
        statusElement.textContent = 'Detection error: ' + error.message;
    }
    
    // Continue the detection loop
    rafId = requestAnimationFrame(detectPoses);
}

// Process detected poses to count punches
function processPoses(poses) {
    if (!poses || poses.length === 0) return;
    
    const pose = poses[0]; // Get the first detected person
    if (!pose || !pose.keypoints) return;
    
    // Get relevant keypoints
    const leftWrist = getKeypoint(pose, 'left_wrist');
    const leftShoulder = getKeypoint(pose, 'left_shoulder');
    const leftElbow = getKeypoint(pose, 'left_elbow');
    const rightWrist = getKeypoint(pose, 'right_wrist');
    const rightShoulder = getKeypoint(pose, 'right_shoulder');
    const rightElbow = getKeypoint(pose, 'right_elbow');
    
    // Track how long wrists have been in frame with good confidence score
    updateWristFrameTracking(leftWrist, rightWrist);
    
    // Check if we have the necessary keypoints
    if (!leftWrist || !leftShoulder || !rightWrist || !rightShoulder) {
        return;
    }
    
    // Calculate arm extensions
    const leftArmExtension = calculateArmExtension(leftWrist, leftShoulder);
    const rightArmExtension = calculateArmExtension(rightWrist, rightShoulder);
    
    // Track arm retraction state (must go back to retracted position between punches)
    if (leftArmExtension < ARM_RETRACTION_THRESHOLD) {
        leftArmRetracted = true;
    }
    
    if (rightArmExtension < ARM_RETRACTION_THRESHOLD) {
        rightArmRetracted = true;
    }
    
    // Handle cooldown period
    if (punchState === 'cooldown') {
        cooldownFrames--;
        if (cooldownFrames <= 0) {
            punchState = 'ready';
        }
        return;
    }
    
    // Calculate velocities if we have previous data
    if (lastPoseData) {
        // Calculate left arm movement
        const leftWristVelocity = calculateVelocity(
            leftWrist, 
            lastPoseData.leftWrist
        );
        
        // Calculate right arm movement
        const rightWristVelocity = calculateVelocity(
            rightWrist, 
            lastPoseData.rightWrist
        );
        
        // Check for punches - looking for forward motion primarily
        if (punchState === 'ready') {
            // Check left punch - now also check that arm was retracted before this punch
            if (leftArmRetracted && detectPunch(leftWristVelocity, leftWrist, leftShoulder, leftElbow, 'left')) {
                incrementCounter();
                punchState = 'cooldown';
                cooldownFrames = COOLDOWN_PERIOD;
                leftArmRetracted = false; // Require retraction before next punch
                return;
            }
            
            // Check right punch - now also check that arm was retracted before this punch
            if (rightArmRetracted && detectPunch(rightWristVelocity, rightWrist, rightShoulder, rightElbow, 'right')) {
                incrementCounter();
                punchState = 'cooldown';
                cooldownFrames = COOLDOWN_PERIOD;
                rightArmRetracted = false; // Require retraction before next punch
                return;
            }
        }
    }
    
    // Store current pose data for velocity calculation in the next frame
    lastPoseData = {
        leftWrist,
        leftShoulder,
        leftElbow,
        rightWrist,
        rightShoulder,
        rightElbow
    };
}

// New function to track how long wrists have been in frame
function updateWristFrameTracking(leftWrist, rightWrist) {
    // Update left wrist tracking
    if (leftWrist && leftWrist.score > CONFIDENCE_THRESHOLD && 
        isPointInFrame(leftWrist, 0.1)) {
        leftWristFramesInFrame++;
    } else {
        leftWristFramesInFrame = 0;
    }
    
    // Update right wrist tracking
    if (rightWrist && rightWrist.score > CONFIDENCE_THRESHOLD && 
        isPointInFrame(rightWrist, 0.1)) {
        rightWristFramesInFrame++;
    } else {
        rightWristFramesInFrame = 0;
    }
}

// Check if a point is within the frame with a margin
function isPointInFrame(point, margin) {
    const width = canvasElement.width;
    const height = canvasElement.height;
    
    return point.x >= width * margin && 
           point.x <= width * (1 - margin) && 
           point.y >= height * margin && 
           point.y <= height * (1 - margin);
}

// Helper function to get a specific keypoint from the pose
function getKeypoint(pose, name) {
    // If using MoveNet, keypoints are already in snake_case format
    if (model === poseDetection.SupportedModels.MoveNet) {
        return pose.keypoints.find(kp => kp.name === name);
    }
    
    // PoseNet mapping (keeping for reference in case we switch back)
    const poseNetMapping = {
        'left_shoulder': 'leftShoulder',
        'right_shoulder': 'rightShoulder',
        'left_elbow': 'leftElbow',
        'right_elbow': 'rightElbow',
        'left_wrist': 'leftWrist',
        'right_wrist': 'rightWrist',
        'left_hip': 'leftHip',
        'right_hip': 'rightHip',
        'left_knee': 'leftKnee',
        'right_knee': 'rightKnee',
        'left_ankle': 'leftAnkle',
        'right_ankle': 'rightAnkle',
        'left_eye': 'leftEye',
        'right_eye': 'rightEye',
        'left_ear': 'leftEar',
        'right_ear': 'rightEar',
        'nose': 'nose'
    };
    
    // For PoseNet
    const lookupName = poseNetMapping[name] || name;
    return pose.keypoints.find(kp => kp.name === lookupName);
}

// Calculate velocity between two points across frames
function calculateVelocity(current, previous) {
    if (!current || !previous) return { x: 0, y: 0, magnitude: 0 };
    
    const xVel = current.x - previous.x;
    const yVel = current.y - previous.y;
    const magnitude = Math.sqrt(xVel * xVel + yVel * yVel);
    
    return {
        x: xVel,
        y: yVel,
        magnitude
    };
}

// Detect if a movement is a punch
function detectPunch(velocity, wrist, shoulder, elbow, side) {
    if (!velocity || !wrist || !shoulder || !elbow) return false;
    
    // Less strict requirement for wrist being in frame
    if (wrist.score < CONFIDENCE_THRESHOLD || 
        !isPointInFrame(wrist, 0.1) || 
        (side === 'left' && leftWristFramesInFrame < MIN_FRAMES_IN_FRAME) ||
        (side === 'right' && rightWristFramesInFrame < MIN_FRAMES_IN_FRAME)) {
        
        // Update debug info
        if (side === 'left') {
            debugInfo.leftPunchDirection = null;
        } else {
            debugInfo.rightPunchDirection = null;
        }
        
        return false;
    }
    
    // Check if the velocity is high enough
    if (velocity.magnitude < VELOCITY_THRESHOLD) {
        // Update debug info
        if (side === 'left') {
            debugInfo.leftPunchDirection = 'too slow';
        } else {
            debugInfo.rightPunchDirection = 'too slow';
        }
        
        return false;
    }
    
    // Check if the arm is extended (wrist is far from shoulder)
    const armExtension = Math.sqrt(
        Math.pow(wrist.x - shoulder.x, 2) + 
        Math.pow(wrist.y - shoulder.y, 2)
    );
    
    // Normalize by using the canvas dimensions
    const normalizedExtension = armExtension / Math.sqrt(
        Math.pow(canvasElement.width, 2) + 
        Math.pow(canvasElement.height, 2)
    );
    
    // Only count punches with proper extension
    if (normalizedExtension < EXTENSION_THRESHOLD) {
        // Update debug info
        if (side === 'left') {
            debugInfo.leftPunchDirection = 'arm not extended';
        } else {
            debugInfo.rightPunchDirection = 'arm not extended';
        }
        
        return false;
    }
    
    // Check for forward motion using multiple methods
    let isForwardMotion = false;
    
    // First method: Check if velocity is more in the Z direction using 2D projection
    // When moving toward the camera, the wrist Y position should move down relative to the elbow
    if (lastPoseData) {
        const prevWrist = lastPoseData[side === 'left' ? 'leftWrist' : 'rightWrist'];
        const prevElbow = lastPoseData[side === 'left' ? 'leftElbow' : 'rightElbow'];
        const prevShoulder = lastPoseData[side === 'left' ? 'leftShoulder' : 'rightShoulder'];
        
        if (prevWrist && prevElbow && prevShoulder) {
            // METHOD 1: Check if wrist is getting closer to the camera relative to elbow
            // We can estimate this by checking if the wrist's Y position is increasing faster than the elbow's
            const wristYChange = wrist.y - prevWrist.y;
            const elbowYChange = elbow.y - prevElbow.y;
            
            // If wrist is moving down faster than elbow, it's moving toward camera
            // (in the canvas Y increases downward)
            const isMovingTowardCamera = wristYChange > elbowYChange + 3; // Added small threshold
            
            // METHOD 2: Check if arm length appears to be increasing (punch toward camera)
            const prevWristElbowDist = Math.sqrt(
                Math.pow(prevWrist.x - prevElbow.x, 2) + 
                Math.pow(prevWrist.y - prevElbow.y, 2)
            );
            
            const currWristElbowDist = Math.sqrt(
                Math.pow(wrist.x - elbow.x, 2) + 
                Math.pow(wrist.y - elbow.y, 2)
            );
            
            // If the apparent length is increasing, arm is moving toward camera
            const isArmLengthIncreasing = currWristElbowDist > prevWristElbowDist * 1.05;
            
            // METHOD 3: Check if fist is moving more than elbow (punching motion)
            const wristMovement = Math.sqrt(
                Math.pow(wrist.x - prevWrist.x, 2) + 
                Math.pow(wrist.y - prevWrist.y, 2)
            );
            
            const elbowMovement = Math.sqrt(
                Math.pow(elbow.x - prevElbow.x, 2) + 
                Math.pow(elbow.y - prevElbow.y, 2)
            );
            
            const isWristMovingMoreThanElbow = wristMovement > elbowMovement * 1.2;
            
            // METHOD 4: Check if arm is aligned toward the camera
            // For a straight punch, the wrist should be generally aligned with the elbow and shoulder
            // in terms of x-coordinate (horizontally aligned in front of body)
            const isWristAlignedWithShoulder = Math.abs(wrist.x - shoulder.x) < 
                                              canvasElement.width * 0.15; // Wrist should be somewhat aligned with shoulder
            
            // METHOD 5: Check for z-axis movement by analyzing how the wrist-to-shoulder ratio changes
            // A forward punch will cause the wrist to move toward the camera, changing its apparent position
            const prevWristShoulderDist = Math.sqrt(
                Math.pow(prevWrist.x - prevShoulder.x, 2) + 
                Math.pow(prevWrist.y - prevShoulder.y, 2)
            );
            
            const currWristShoulderDist = Math.sqrt(
                Math.pow(wrist.x - shoulder.x, 2) + 
                Math.pow(wrist.y - shoulder.y, 2)
            );
            
            // If this ratio increases significantly, it suggests forward movement
            const wristToShoulderRatioChange = currWristShoulderDist / prevWristShoulderDist;
            const isWristMovingForward = wristToShoulderRatioChange > 1.03;
            
            // Combine methods - now require at least 2 of these conditions to be true 
            // for a punch to be considered toward the front
            const methodsTrue = [
                isMovingTowardCamera, 
                isArmLengthIncreasing, 
                isWristMovingMoreThanElbow,
                isWristAlignedWithShoulder,
                isWristMovingForward
            ].filter(Boolean).length;
            
            // More strict: now require at least 2 conditions to be true
            isForwardMotion = methodsTrue >= 2;
            
            // Log detailed information but limit frequency
            if (Date.now() - debugInfo.lastCheck > 300) {
                console.log(`${side} punch: Forward=${isForwardMotion}, ToCamera=${isMovingTowardCamera}, ArmLen=${isArmLengthIncreasing}, WristMovement=${isWristMovingMoreThanElbow}, Aligned=${isWristAlignedWithShoulder}, ZMovement=${isWristMovingForward}`);
                debugInfo.lastCheck = Date.now();
            }
            
            // Update debug info for visualization
            if (side === 'left') {
                debugInfo.leftPunchDirection = isForwardMotion ? 'forward' : 'sideways';
            } else {
                debugInfo.rightPunchDirection = isForwardMotion ? 'forward' : 'sideways';
            }
        }
    }
    
    if (!isForwardMotion) {
        return false;
    }
    
    // If all checks pass, this is a valid punch
    return true;
}

// Increment the punch counter
function incrementCounter() {
    punchCount++;
    counterElement.textContent = punchCount;
    
    // Add visual feedback for the punch
    canvasElement.style.border = '5px solid #e74c3c';
    
    // Add flash effect
    const overlay = document.createElement('div');
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(231, 76, 60, 0.3)';
    overlay.style.zIndex = '10';
    overlay.style.pointerEvents = 'none';
    document.body.appendChild(overlay);
    
    setTimeout(() => {
        canvasElement.style.border = '';
        document.body.removeChild(overlay);
    }, 150);
}

// Draw detected poses and additional information
function drawResults(poses) {
    // Clear the canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (!poses || poses.length === 0) return;
    
    // Draw keypoints and connections
    poses.forEach(pose => {
        if (pose.keypoints) {
            drawKeypoints(pose.keypoints);
            drawSkeleton(pose.keypoints);
        }
    });
}

// Draw keypoints on the canvas
function drawKeypoints(keypoints) {
    keypoints.forEach(keypoint => {
        // Only draw keypoints with a high enough confidence
        if (keypoint.score > 0.3) {
            const { x, y } = keypoint;
            
            // Use white color for all keypoints
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

// Draw skeleton (connections between keypoints)
function drawSkeleton(keypoints) {
    // Define the skeleton connections
    const adjacentKeyPoints = [
        ['nose', 'left_eye'], ['nose', 'right_eye'],
        ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
        ['nose', 'left_shoulder'], ['nose', 'right_shoulder'],
        ['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow'],
        ['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist'],
        ['left_shoulder', 'right_shoulder'],
        ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
        ['left_hip', 'right_hip'],
        ['left_hip', 'left_knee'], ['right_hip', 'right_knee'],
        ['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']
    ];
    
    // Convert keypoint names for PoseNet if needed
    const poseNetMapping = {
        'left_shoulder': 'leftShoulder',
        'right_shoulder': 'rightShoulder',
        'left_elbow': 'leftElbow',
        'right_elbow': 'rightElbow',
        'left_wrist': 'leftWrist',
        'right_wrist': 'rightWrist',
        'left_hip': 'leftHip',
        'right_hip': 'rightHip',
        'left_knee': 'leftKnee',
        'right_knee': 'rightKnee',
        'left_ankle': 'leftAnkle',
        'right_ankle': 'rightAnkle',
        'left_eye': 'leftEye',
        'right_eye': 'rightEye',
        'left_ear': 'leftEar',
        'right_ear': 'rightEar',
        'nose': 'nose'
    };
    
    // Use plain white color for all skeleton lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 2;
    
    // Draw all connections with consistent style
    adjacentKeyPoints.forEach(([firstKpName, secondKpName]) => {
        // Get keypoints with proper naming for the current model
        let firstName = firstKpName;
        let secondName = secondKpName;
        
        // Only map for PoseNet
        if (model === poseDetection.SupportedModels.PoseNet) {
            firstName = poseNetMapping[firstKpName] || firstKpName;
            secondName = poseNetMapping[secondKpName] || secondKpName;
        }
        
        const firstKeypoint = keypoints.find(kp => kp.name === firstName);
        const secondKeypoint = keypoints.find(kp => kp.name === secondName);
        
        if (firstKeypoint && secondKeypoint && 
            firstKeypoint.score > 0.3 && secondKeypoint.score > 0.3) {
            ctx.beginPath();
            ctx.moveTo(firstKeypoint.x, firstKeypoint.y);
            ctx.lineTo(secondKeypoint.x, secondKeypoint.y);
            ctx.stroke();
        }
    });
}

// Calculate arm extension (normalized)
function calculateArmExtension(wrist, shoulder) {
    if (!wrist || !shoulder) return null;
    
    const extension = Math.sqrt(
        Math.pow(wrist.x - shoulder.x, 2) + 
        Math.pow(wrist.y - shoulder.y, 2)
    );
    
    return extension / Math.sqrt(
        Math.pow(canvasElement.width, 2) + 
        Math.pow(canvasElement.height, 2)
    );
}

// Start the punch counter application
async function startApp() {
    if (isRunning) return;
    
    try {
        // Update UI
        startButton.disabled = true;
        statusElement.textContent = 'Starting camera...';
        
        // Initialize camera
        await startCamera();
        
        // Set canvas dimensions to match video
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        
        // Log video dimensions for debugging
        console.log('Video dimensions:', videoElement.videoWidth, 'x', videoElement.videoHeight);
        
        // Start pose detection
        isRunning = true;
        statusElement.textContent = 'Punch counter active';
        resetButton.disabled = false;
        stopButton.disabled = false;
        
        detectPoses();
    } catch (error) {
        console.error('Error starting the application:', error);
        statusElement.textContent = 'Error: ' + error.message;
        startButton.disabled = false;
    }
}

// Reset the punch counter
function resetCounter() {
    punchCount = 0;
    counterElement.textContent = punchCount;
    
    // Reset punch states
    punchState = 'ready';
    cooldownFrames = 0;
    leftArmRetracted = true;
    rightArmRetracted = true;
    
    // Add visual feedback
    const overlay = document.createElement('div');
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(52, 152, 219, 0.3)'; // Blue flash
    overlay.style.zIndex = '10';
    overlay.style.pointerEvents = 'none';
    document.body.appendChild(overlay);
    
    // Remove flash effect after a short delay
    setTimeout(() => {
        document.body.removeChild(overlay);
    }, 150);
    
    console.log('Counter reset to 0');
}

// Stop the application
function stopApp() {
    if (!isRunning) return;
    
    // Stop the detection loop
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }
    
    // Stop the webcam
    if (videoElement.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    
    // Reset variables
    lastPoseData = null;
    punchState = 'ready';
    cooldownFrames = 0;
    
    // Update UI
    isRunning = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    resetButton.disabled = true;
    statusElement.textContent = 'Application stopped';
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

// Event listeners
startButton.addEventListener('click', startApp);
stopButton.addEventListener('click', stopApp);
resetButton.addEventListener('click', resetCounter);

// Initialize the application when the page loads
window.addEventListener('load', init); 