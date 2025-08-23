"""
Blink-PIN Authentication System (Fixed MediaPipe version)

This script uses MediaPipe's face mesh to detect blinks and convert them to a PIN.
The EAR calculation has been fixed to work properly with MediaPipe landmarks.

Requirements:
- Python 3.10-3.12
- OpenCV (opencv-python)
- MediaPipe
- NumPy

Installation:
pip install opencv-python mediapipe numpy
"""

import cv2
import time
import numpy as np
import hashlib
import mediapipe as mp

# -------------------
# CONFIGURATION
# -------------------
PIN = "0101"  
MAX_BLINKS = 4

# Thresholds (properly calibrated)
EAR_THRESHOLD = 0.25  # Standard threshold for blink detection
BLINK_DURATION_THRESHOLD = 0.4  # Quick vs Long blink
MIN_BLINK_INTERVAL = 0.5  # Minimum time between blinks
CONSEC_FRAMES = 3  # Consecutive frames to confirm blink

# Mappings
BLINK_TO_DIGIT = {
    "quick": "0",
    "long": "1"
}

# -------------------
# HELPER FUNCTIONS
# -------------------

def hash_pin(pin):
    """Hash PIN using SHA-256"""
    return hashlib.sha256(pin.encode()).hexdigest()

def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio using the standard formula.
    eye_points should be 6 landmarks in this order:
    [0] = outer corner
    [1] = top point 1 
    [2] = top point 2
    [3] = inner corner  
    [4] = bottom point 1
    [5] = bottom point 2
    """
    # Convert to numpy array
    points = np.array(eye_points, dtype=np.float32)
    
    # Calculate the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(points[1] - points[5])  # Vertical distance 1
    B = np.linalg.norm(points[2] - points[4])  # Vertical distance 2
    
    # Calculate the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(points[0] - points[3])  # Horizontal distance
    
    # Calculate the eye aspect ratio
    if C == 0:
        return 0
    
    ear = (A + B) / (2.0 * C)
    return ear

def get_landmark_coords(landmarks, indices, width, height):
    """Extract landmark coordinates and convert to pixels"""
    coords = []
    for idx in indices:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        coords.append([x, y])
    return coords

# -------------------
# MEDIAPIPE SETUP
# -------------------

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Correct MediaPipe eye landmark indices for EAR calculation
# Left eye: outer corner, top, top, inner corner, bottom, bottom
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # 6 points for left eye EAR
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # 6 points for right eye EAR

# Initialize face mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------
# MAIN PROGRAM
# -------------------

print("[INFO] Initializing Blink-PIN Authentication System")
print(f"[INFO] Target PIN: {PIN}")
print("[INFO] Quick blink = 0, Long blink = 1")
print("[INFO] Press 'q' to quit, 'r' to reset")

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access camera")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State variables
blink_sequence = []
frame_counter = 0
ear_history = []
is_blinking = False
blink_start_time = 0
last_blink_time = 0
consec_blinks = 0

print("[INFO] Camera initialized. Starting detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break
    
    frame_counter += 1
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = face_mesh.process(rgb_frame)
    
    current_time = time.time()
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye coordinates
            left_eye = get_landmark_coords(face_landmarks, LEFT_EYE_LANDMARKS, w, h)
            right_eye = get_landmark_coords(face_landmarks, RIGHT_EYE_LANDMARKS, w, h)
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            # Average both eyes
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Store EAR history for smoothing
            ear_history.append(avg_ear)
            if len(ear_history) > 5:
                ear_history.pop(0)
            
            # Use smoothed EAR
            smooth_ear = np.mean(ear_history) if ear_history else avg_ear
            
            # Draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            
            # Draw eye contours
            left_contour = np.array(left_eye, np.int32)
            right_contour = np.array(right_eye, np.int32)
            cv2.polylines(frame, [left_contour], True, (255, 0, 0), 1)
            cv2.polylines(frame, [right_contour], True, (255, 0, 0), 1)
            
            # Blink detection
            if smooth_ear < EAR_THRESHOLD:
                consec_blinks += 1
                
                if not is_blinking and consec_blinks >= CONSEC_FRAMES:
                    if (current_time - last_blink_time) > MIN_BLINK_INTERVAL:
                        is_blinking = True
                        blink_start_time = current_time
                        print(f"[BLINK START] EAR: {smooth_ear:.3f}")
                        
                        # Visual feedback
                        cv2.putText(frame, "BLINK DETECTED!", (50, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                consec_blinks = 0
                
                if is_blinking:
                    blink_duration = current_time - blink_start_time
                    
                    if len(blink_sequence) < MAX_BLINKS:
                        if blink_duration < BLINK_DURATION_THRESHOLD:
                            blink_type = "quick"
                            digit = "0"
                            color = (0, 255, 0)
                        else:
                            blink_type = "long"
                            digit = "1"
                            color = (0, 0, 255)
                        
                        blink_sequence.append(blink_type)
                        last_blink_time = current_time
                        
                        print(f"[DETECTED] {blink_type.upper()} blink ({blink_duration:.2f}s) -> {digit}")
                        
                        # Show detection on screen
                        cv2.putText(frame, f"{blink_type.upper()} -> {digit}", (50, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    is_blinking = False
            
            # Display EAR and status
            cv2.putText(frame, f"EAR: {smooth_ear:.3f}", (w-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (w-150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Debug info every 60 frames
            if frame_counter % 60 == 0:
                print(f"[DEBUG] Current EAR: {smooth_ear:.3f} (Threshold: {EAR_THRESHOLD})")
    
    else:
        cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display progress
    progress_text = f"PIN: {len(blink_sequence)}/{MAX_BLINKS}"
    cv2.putText(frame, progress_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Display current sequence
    if blink_sequence:
        sequence = "".join([BLINK_TO_DIGIT[b] for b in blink_sequence])
        cv2.putText(frame, f"Sequence: {sequence}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Instructions
    cv2.putText(frame, "Quick=0, Long=1 | Q=Quit, R=Reset", (10, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Show frame
    cv2.imshow("Blink-PIN Authentication", frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("[INFO] Resetting sequence...")
        blink_sequence = []
        is_blinking = False
        consec_blinks = 0
    
    # Check completion
    if len(blink_sequence) >= MAX_BLINKS:
        print("[INFO] PIN entry complete!")
        break

# -------------------
# CLEANUP & VALIDATION
# -------------------

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# Validate PIN
if len(blink_sequence) >= MAX_BLINKS:
    entered_pin = "".join([BLINK_TO_DIGIT[b] for b in blink_sequence])
    
    print(f"\nBlink sequence: {blink_sequence}")
    print(f"Entered PIN: {entered_pin}")
    print(f"Expected PIN: {PIN}")
    
    if hash_pin(entered_pin) == hash_pin(PIN):
        print("\n" + "="*35)
        print("   🎉 AUTHENTICATION SUCCESS! 🎉")
        print("="*35)
    else:
        print("\n" + "="*35)
        print("   ❌ AUTHENTICATION FAILED ❌")
        print("="*35)
else:
    print("\n[INFO] PIN entry incomplete")

print(f"\nSession Summary:")
print(f"- Blinks detected: {len(blink_sequence)}")
print(f"- EAR threshold: {EAR_THRESHOLD}")
print(f"- Duration threshold: {BLINK_DURATION_THRESHOLD}s")
