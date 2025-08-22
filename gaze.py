import cv2
import numpy as np
import pyautogui
import time
from collections import deque
import mediapipe as mp
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
USE_FULLSCREEN_CALIB_UI = True

# Training strategy options:
# "ALWAYS" - Retrain after every calibration (best accuracy, slower)
# "INCREMENTAL" - Continuously improve model with new samples (recommended)
# "THRESHOLD" - Retrain only when reaching sample threshold (balanced)
# "MANUAL" - Only retrain when pressing 't' key (fastest, manual control)
TRAINING_MODE = "INCREMENTAL"  
RETRAIN_THRESHOLD = 40  # Reduced threshold for more frequent training
INCREMENTAL_TRAINING_SAMPLES = 15  # Reduced for more frequent incremental training

# Position change detection and adaptation - enhanced sensitivity
POSITION_CHANGE_THRESHOLD = 0.08  # Reduced threshold for more sensitive position change detection
BASELINE_UPDATE_SAMPLES = 12      # Increased samples for more accurate baseline update
AUTO_RECALIB_ON_POSITION_CHANGE = True  # Auto-recalibrate when position changes detected
HEAD_POSE_CHANGE_THRESHOLD = 8.0  # Reduced threshold for more sensitive head pose change detection
AUTO_ADAPTATION_MODE = True       # Automatically adapt using previous training data
ADAPTATION_CONFIDENCE_THRESHOLD = 0.7  # Increased threshold for more aggressive adaptation

# Enhanced calibration grid - IMPROVED TOP AREA ACCURACY
# Extra emphasis on top portion with more calibration points
CALIB_POINTS_GRID = [
    # 1. CENTER POINT FIRST - establish baseline
    (0.5, 0.5),
    
    # 2. TOP AREA INTENSIVE COVERAGE (extra points for better accuracy)
    (0.5, 0.02),   # very top center
    (0.25, 0.05), (0.75, 0.05),  # top quarter points
    (0.1, 0.08), (0.9, 0.08),    # top edges
    (0.35, 0.08), (0.65, 0.08),  # top intermediate
    (0.15, 0.12), (0.85, 0.12),  # top region
    (0.2, 0.05), (0.4, 0.05), (0.6, 0.05), (0.8, 0.05),  # top fine grid
    
    # 3. EXTREME CORNERS - map full range
    (0.02, 0.02), (0.98, 0.02), (0.02, 0.98), (0.98, 0.98),
    
    # 4. UPPER REGION ENHANCEMENT
    (0.12, 0.18), (0.25, 0.18), (0.5, 0.18), (0.75, 0.18), (0.88, 0.18),
    (0.08, 0.25), (0.3, 0.25), (0.7, 0.25), (0.92, 0.25),
    
    # 5. EDGE MIDPOINTS - establish boundaries  
    (0.5, 0.95),   # bottom center
    (0.05, 0.5),   # left center
    (0.95, 0.5),   # right center
    
    # 6. REGULAR GRID - good coverage for middle and bottom
    # Upper-middle row  
    (0.15, 0.35), (0.35, 0.35), (0.65, 0.35), (0.85, 0.35),
    # Middle row
    (0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5),
    # Lower-middle row
    (0.15, 0.65), (0.35, 0.65), (0.65, 0.65), (0.85, 0.65),
    # Bottom row
    (0.15, 0.85), (0.35, 0.85), (0.65, 0.85), (0.85, 0.85),
    
    # 7. ADDITIONAL TOP-SPECIFIC POINTS
    (0.05, 0.15), (0.95, 0.15),  # top side points
    (0.3, 0.12), (0.7, 0.12),    # top intermediate
    (0.45, 0.08), (0.55, 0.08),  # top center variants
    
    # 8. FINE EDGE COVERAGE
    (0.1, 0.2), (0.1, 0.4), (0.1, 0.6), (0.1, 0.8),    # left fine  
    (0.9, 0.2), (0.9, 0.4), (0.9, 0.6), (0.9, 0.8),    # right fine
    (0.2, 0.9), (0.4, 0.9), (0.6, 0.9), (0.8, 0.9),    # bottom fine
]

SAMPLES_PER_POINT = 12            # Increased samples for better accuracy per calibration point
BASELINE_SAMPLES = 20             # Increased samples for more accurate center point baseline
TOP_AREA_SAMPLES = 15             # Increased samples for better top area accuracy (Y < 0.3)
STABILITY_WINDOW = 7              # Increased window for better stability assessment
STABILITY_STD_MAX = 0.008         # Stricter stability threshold for more accurate calibration
TOP_AREA_STABILITY_MAX = 0.012    # Stricter threshold while still accommodating top area challenges
SMOOTHING_WINDOW = 6              # Increased smoothing for more stable cursor movement
BLINK_EAR_THRESH = 0.19           # Adjusted threshold for more reliable blink detection
BLINK_MIN_FRAMES = 2              # Reduced frames for faster click response
WINK_RIGHT_DIFF = 0.07            # Fine-tuned threshold for more accurate wink detection
WINK_MIN_FRAMES = 3               # Reduced frames for faster right-click response
SHOW_DEBUG_OVERLAY = True           # press 'v' to toggle
RECALIB_KEY = ord('c')
QUIT_KEYS = {27, ord('q')}          # ESC or q

# Model configuration
MODEL_DIR = "gaze_models"
MODEL_FILE_X = os.path.join(MODEL_DIR, "gaze_model_x.pkl")
MODEL_FILE_Y = os.path.join(MODEL_DIR, "gaze_model_y.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "feature_scaler.pkl")
DATA_FILE = os.path.join(MODEL_DIR, "training_data.pkl")
USE_NEURAL_NETWORK = True  # Set to False to use Random Forest

# Training strategy options
TRAINING_MODE = "ALWAYS"  # Options: "ALWAYS", "THRESHOLD", "MANUAL"
RETRAIN_THRESHOLD = 100  # Only used if TRAINING_MODE = "THRESHOLD"
AUTO_SAVE_FREQUENCY = 50  # Auto-save training data every N samples

EXPECTED_FEATURE_DIM = 6  # Must match the length of feature vectors used throughout

# =========================
# INIT
# =========================
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,          # required for iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices (MediaPipe FaceMesh)
# Eyes (key points)
L_OUT, L_IN = 33, 133
R_OUT, R_IN = 263, 362
# Eyelid tops/bottoms for EAR
L_UP = [159, 158]; L_DN = [145, 153]
R_UP = [386, 385]; R_DN = [374, 380]
# Iris
L_IRIS = [468, 469, 470, 471, 472]
R_IRIS = [473, 474, 475, 476, 477]

# Head pose keypoints (2D->3D mapping using a generic model)
# Using commonly stable points with additional landmarks for better accuracy
POSE_LANDMARKS = {
    "nose_tip": 1,      # approximate
    "chin": 152,
    "l_eye_outer": 33,
    "r_eye_outer": 263,
    "l_mouth": 61,
    "r_mouth": 291,
    "forehead": 10,    # Top of forehead
    "l_eye_inner": 133,
    "r_eye_inner": 362
}
# Enhanced 3D model points with more landmarks for better pose estimation
# Units are arbitrary; relative distances matter for accurate pose estimation
MODEL_3D_POINTS = np.array([
    [0.0,   0.0,   0.0],    # nose tip
    [0.0,  -85.0,  -5.0],   # chin
    [-40.0, 35.0, -30.0],   # left eye outer
    [ 40.0, 35.0, -30.0],   # right eye outer
    [-28.0,-30.0, -30.0],   # left mouth
    [ 28.0,-30.0, -30.0],   # right mouth
    [0.0,   70.0, -5.0],    # forehead
    [-20.0, 35.0, -35.0],   # left eye inner
    [ 20.0, 35.0, -35.0]    # right eye inner
], dtype=np.float64)

def ear_ratio(landmarks, idx_up, idx_dn, idx_left, idx_right, w, h):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||), adapted to FaceMesh indices
    pu = np.mean([[landmarks[i].x*w, landmarks[i].y*h] for i in idx_up], axis=0)
    pd = np.mean([[landmarks[i].x*w, landmarks[i].y*h] for i in idx_dn], axis=0)
    pl = np.array([landmarks[idx_left].x*w, landmarks[idx_left].y*h])
    pr = np.array([landmarks[idx_right].x*w, landmarks[idx_right].y*h])
    v = np.linalg.norm(pu - pd)
    hdist = np.linalg.norm(pl - pr)
    if hdist < 1e-6: 
        return 0.0
    return (2.0 * v) / (2.0 * hdist)

def iris_center(landmarks, iris_idx, w, h):
    pts = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in iris_idx], dtype=np.float32)
    return np.mean(pts, axis=0)

def eye_box_norm(landmarks, left_idx, right_idx, w, h, pt):
    # Enhanced normalization for better upward gaze tracking
    pL = np.array([landmarks[left_idx].x*w, landmarks[left_idx].y*h])
    pR = np.array([landmarks[right_idx].x*w, landmarks[right_idx].y*h])
    ex = (pt[0] - pL[0]) / max(np.linalg.norm(pR - pL), 1e-6)
    
    # Improved vertical normalization for better top area accuracy
    if left_idx == L_OUT:  # Left eye
        top = np.mean([[landmarks[i].x*w, landmarks[i].y*h] for i in L_UP], axis=0)
        bot = np.mean([[landmarks[i].x*w, landmarks[i].y*h] for i in L_DN], axis=0)
    else:  # Right eye
        top = np.mean([[landmarks[i].x*w, landmarks[i].y*h] for i in R_UP], axis=0)
        bot = np.mean([[landmarks[i].x*w, landmarks[i].y*h] for i in R_DN], axis=0)
    
    # Enhanced vertical calculation with extended range for upward gaze
    eye_height = max(np.linalg.norm(bot - top), 1e-6)
    
    # Calculate vertical position with better sensitivity for upward movement
    basic_ez = (pt[1] - top[1]) / eye_height
    
    # Apply non-linear transformation to improve top area sensitivity
    # Enhanced non-linear transformation for better accuracy across all gaze directions
    if basic_ez < 0:  # Looking up (iris above top lid)
        # Stronger exponential scaling for upward gaze with improved precision
        ez = basic_ez * 2.2  # Significantly increased sensitivity for upward gaze
    elif basic_ez > 1:  # Looking down (iris below bottom lid)  
        ez = 1 + (basic_ez - 1) * 1.5  # Enhanced expansion for downward gaze
    else:
        # Normal range (0-1), apply optimized bias for center-region accuracy
        # Apply quadratic transformation for better mid-range precision
        normalized = (basic_ez - 0.5) * 2  # Scale to [-1, 1]
        adjusted = 0.5 + (normalized * (1.0 - 0.1 * normalized * normalized)) / 2
        ez = adjusted * 0.98 + 0.01  # Fine-tuned shift
    
    return np.array([ex, ez], dtype=np.float32)

def head_pose_yaw_pitch(landmarks, w, h, frame_w, frame_h):
    # Use only the original 6 landmarks for stability
    idx = [POSE_LANDMARKS["nose_tip"], POSE_LANDMARKS["chin"], POSE_LANDMARKS["l_eye_outer"],
           POSE_LANDMARKS["r_eye_outer"], POSE_LANDMARKS["l_mouth"], POSE_LANDMARKS["r_mouth"]]
    
    # Convert to 2D points with improved precision
    pts_2d = np.array([[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in idx], dtype=np.float64)
    
    # Enhanced camera matrix with better focal length estimation
    # Using a more realistic focal length estimation based on typical webcam FOV
    focal = frame_w * 1.2  # Adjusted focal length for better depth perception
    cam_mat = np.array([[focal, 0, frame_w/2],
                        [0, focal, frame_h/2],
                        [0, 0, 1]], dtype=np.float64)
    
    # Add slight distortion correction for typical webcams
    dist = np.array([0.1, -0.1, 0, 0], dtype=np.float64).reshape(4,1)
    
    # Use original 3D points (first 6 points) for stability
    model_points = MODEL_3D_POINTS[:6]
    
    # Use EPNP algorithm which is more robust for planar configurations
    success, rvec, tvec = cv2.solvePnP(model_points, pts_2d, cam_mat, dist, flags=cv2.SOLVEPNP_EPNP)
    
    # Refine the pose for higher accuracy
    if success:
        success, rvec, tvec = cv2.solvePnP(model_points, pts_2d, cam_mat, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return 0.0, 0.0
        
    R, _ = cv2.Rodrigues(rvec)
    
    # Extract yaw (y-axis), pitch (x-axis) with improved calculation
    sy = -R[2,0]
    cy = np.sqrt(1 - sy*sy)
    yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))
    pitch = np.degrees(np.arctan2(-R[2,0], cy + 1e-6))
    
    # Apply smoothing to reduce jitter
    yaw = max(min(yaw, 90), -90)  # Limit to reasonable range
    pitch = max(min(pitch, 90), -90)  # Limit to reasonable range
    
    return float(yaw), float(pitch)

# Regression helper (least squares) to map features -> screen x or y
def fit_regression(X, Y):
    # X: N x F, Y: N
    # Add bias column
    Xb = np.hstack([X, np.ones((X.shape[0],1), dtype=np.float64)])
    # Solve (Xb^T Xb) w = Xb^T Y
    w, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    return w

def predict_regression(w, feats):
    xb = np.append(feats, 1.0)
    return float(np.dot(xb, w))

# =========================
# MACHINE LEARNING MODEL FUNCTIONS
# =========================

class GazeModel:
    def __init__(self):
        self.model_x = None
        self.model_y = None
        self.scaler = StandardScaler()
        self.training_data = {'features': [], 'targets_x': [], 'targets_y': []}
        self.baseline_features = None  # Store center point baseline
        self.baseline_screen_pos = None  # Store center screen position
        self.calibration_session_id = 0  # Track calibration sessions
        
        # Position change detection and auto-adaptation
        self.current_center_samples = deque(maxlen=30)  # Recent center-looking samples
        self.position_stable = True
        self.last_position_check = time.time()
        self.baseline_confidence = 0.0  # Confidence in current baseline (0-1)
        self.adaptation_in_progress = False
        self.adaptation_samples = deque(maxlen=15)
        self.last_auto_adaptation = 0
        self.position_drift_buffer = deque(maxlen=50)  # Track position drift over time
        
        # Enhanced training tracking
        self.last_training_sample_count = 0
        self.training_history = []  # Track MSE over time
        self.incremental_samples_since_training = 0
        
        self.feature_dim = EXPECTED_FEATURE_DIM
        self.load_models()
    
    def create_model(self):
        """Create a new model instance optimized for head pose and eye movement tracking"""
        if USE_NEURAL_NETWORK:
            # Enhanced network architecture with better head pose and eye movement integration
            return MLPRegressor(
                hidden_layer_sizes=(384, 192, 96),  # Wider network for better feature extraction
                activation='relu',  # Changed to relu for better performance with head pose data
                solver='adam',
                alpha=0.0003,  # Optimized regularization for head movements
                beta_1=0.9, beta_2=0.999,  # Optimized momentum parameters
                learning_rate='adaptive',
                learning_rate_init=0.001,  # Slightly higher learning rate for faster adaptation
                max_iter=1500,  # Fewer iterations for faster training
                random_state=42,
                early_stopping=True,
                validation_fraction=0.12,  # Balanced validation set
                n_iter_no_change=15,  # Less patience for faster training
                tol=1e-4,  # Slightly relaxed convergence criteria for faster training
                batch_size=min(32, max(16, len(self.training_data['features']) // 10))  # Dynamic batch size
            )
        else:
            # Enhanced forest with better parameter tuning for head pose integration
            return RandomForestRegressor(
                n_estimators=150,  # Fewer trees for faster training
                max_depth=20,  # Balanced tree depth
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                max_features=0.8, # Use 80% of features for better head pose integration
                bootstrap=True,
                criterion='squared_error'
            )
    
    def save_models(self):
        """Save trained models and data to disk"""
        try:
            if self.model_x is not None:
                with open(MODEL_FILE_X, 'wb') as f:
                    pickle.dump(self.model_x, f)
            if self.model_y is not None:
                with open(MODEL_FILE_Y, 'wb') as f:
                    pickle.dump(self.model_y, f)
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save enhanced training data with baseline info
            enhanced_data = {
                'training_data': self.training_data,
                'baseline_features': self.baseline_features,
                'baseline_screen_pos': self.baseline_screen_pos,
                'calibration_session_id': self.calibration_session_id
            }
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(enhanced_data, f)
            print(f"Models saved successfully. Training samples: {len(self.training_data['features'])}")
            if self.baseline_features is not None:
                print(f"Baseline established: {self.baseline_features[:4]} -> {self.baseline_screen_pos}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load previously trained models and data from disk"""
        try:
            # Only load scaler if it matches expected feature dimension
            scaler_loaded = False
            if os.path.exists(SCALER_FILE):
                with open(SCALER_FILE, 'rb') as f:
                    scaler = pickle.load(f)
                    # Check scaler shape
                    if hasattr(scaler, 'mean_') and len(scaler.mean_) == self.feature_dim:
                        self.scaler = scaler
                        scaler_loaded = True
                    else:
                        print(f"Scaler feature dimension mismatch. Ignoring old scaler.")
            if not scaler_loaded:
                self.scaler = StandardScaler()

            # Only load models if scaler is valid (since models depend on feature dim)
            model_x_loaded = False
            if os.path.exists(MODEL_FILE_X):
                with open(MODEL_FILE_X, 'rb') as f:
                    model_x = pickle.load(f)
                    # Check if model expects correct input shape (MLPRegressor/RandomForestRegressor)
                    if hasattr(model_x, 'n_features_in_') and model_x.n_features_in_ == self.feature_dim:
                        self.model_x = model_x
                        model_x_loaded = True
                    else:
                        print(f"Model X feature dimension mismatch. Ignoring old model.")
            if not model_x_loaded:
                self.model_x = None

            model_y_loaded = False
            if os.path.exists(MODEL_FILE_Y):
                with open(MODEL_FILE_Y, 'rb') as f:
                    model_y = pickle.load(f)
                    if hasattr(model_y, 'n_features_in_') and model_y.n_features_in_ == self.feature_dim:
                        self.model_y = model_y
                        model_y_loaded = True
                    else:
                        print(f"Model Y feature dimension mismatch. Ignoring old model.")
            if not model_y_loaded:
                self.model_y = None

            # Load training data and check feature dimension
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'rb') as f:
                    data = pickle.load(f)
                    # Handle both old and new data formats
                    if isinstance(data, dict) and 'training_data' in data:
                        td = data['training_data']
                        # Filter out samples with wrong feature dimension
                        filtered_features = []
                        filtered_targets_x = []
                        filtered_targets_y = []
                        for i, feat in enumerate(td['features']):
                            if len(feat) == self.feature_dim:
                                filtered_features.append(feat)
                                filtered_targets_x.append(td['targets_x'][i])
                                filtered_targets_y.append(td['targets_y'][i])
                        if len(filtered_features) < len(td['features']):
                            print(f"Filtered out {len(td['features']) - len(filtered_features)} samples with wrong feature dimension.")
                        self.training_data = {
                            'features': filtered_features,
                            'targets_x': filtered_targets_x,
                            'targets_y': filtered_targets_y
                        }
                        self.baseline_features = data.get('baseline_features')
                        self.baseline_screen_pos = data.get('baseline_screen_pos')
                        self.calibration_session_id = data.get('calibration_session_id', 0)
                        print(f"Loaded enhanced training data with {len(self.training_data['features'])} samples")
                        if self.baseline_features is not None:
                            print(f"Baseline loaded: {self.baseline_features[:4]} -> {self.baseline_screen_pos}")
                    else:
                        # Old format - just training data
                        td = data
                        filtered_features = []
                        filtered_targets_x = []
                        filtered_targets_y = []
                        for i, feat in enumerate(td['features']):
                            if len(feat) == self.feature_dim:
                                filtered_features.append(feat)
                                filtered_targets_x.append(td['targets_x'][i])
                                filtered_targets_y.append(td['targets_y'][i])
                        if len(filtered_features) < len(td['features']):
                            print(f"Filtered out {len(td['features']) - len(filtered_features)} legacy samples with wrong feature dimension.")
                        self.training_data = {
                            'features': filtered_features,
                            'targets_x': filtered_targets_x,
                            'targets_y': filtered_targets_y
                        }
                        print(f"Loaded legacy training data with {len(self.training_data['features'])} samples")
            else:
                self.training_data = {'features': [], 'targets_x': [], 'targets_y': []}
        except Exception as e:
            print(f"Error loading models: {e}")
            self.training_data = {'features': [], 'targets_x': [], 'targets_y': []}
    
    def set_baseline(self, features, screen_pos):
        """Set the baseline (center point) for relative gaze tracking"""
        self.baseline_features = np.array(features)
        self.baseline_screen_pos = np.array(screen_pos)
        self.baseline_confidence = 1.0  # Full confidence in new baseline
        self.current_center_samples.clear()  # Clear old samples
        print(f"Baseline set: Eye angles {features[:4]} -> Screen {screen_pos}")
    
    def check_position_change(self, current_features, screen_center_tolerance=100):
        """Check if user has significantly changed position/angle"""
        if self.baseline_features is None:
            return False, "No baseline established"
        
        # Calculate feature differences from baseline
        feature_diff = np.abs(current_features[:6] - self.baseline_features[:6])
        
        # Check eye position changes (most sensitive indicator)
        eye_position_change = np.mean(feature_diff[:4])
        
        # Check head pose changes
        head_pose_change = np.sqrt(np.sum(feature_diff[4:6]**2)) * 30.0  # Convert back to degrees
        
        # Determine if change is significant
        position_changed = (eye_position_change > POSITION_CHANGE_THRESHOLD or 
                          head_pose_change > HEAD_POSE_CHANGE_THRESHOLD)
        
        if position_changed:
            # Reduce baseline confidence
            self.baseline_confidence = max(0.0, self.baseline_confidence - 0.1)
            reason = f"Eye pos change: {eye_position_change:.3f}, Head pose change: {head_pose_change:.1f}°"
        else:
            # Increase baseline confidence if stable
            self.baseline_confidence = min(1.0, self.baseline_confidence + 0.02)
            reason = "Position stable"
        
        return position_changed, reason
    
    def update_baseline_adaptively(self, current_features, predicted_screen_pos):
        """Adaptively update baseline when user is looking at center of screen"""
        screen_center_x, screen_center_y = SCREEN_W // 2, SCREEN_H // 2
        
        # Check if predicted position is near screen center
        distance_from_center = np.sqrt((predicted_screen_pos[0] - screen_center_x)**2 + 
                                     (predicted_screen_pos[1] - screen_center_y)**2)
        
        if distance_from_center < 150:  # Within 150 pixels of center
            self.current_center_samples.append(current_features.copy())
            
            # If we have enough samples, update baseline
            if len(self.current_center_samples) >= BASELINE_UPDATE_SAMPLES:
                new_baseline = np.mean(self.current_center_samples, axis=0)
                
                # Only update if the change is significant but not too extreme
                if self.baseline_features is not None:
                    change_magnitude = np.mean(np.abs(new_baseline[:4] - self.baseline_features[:4]))
                    if 0.05 < change_magnitude < 0.3:  # Reasonable change range
                        self.baseline_features = new_baseline
                        self.baseline_screen_pos = np.array([screen_center_x, screen_center_y])
                        self.baseline_confidence = 0.8  # Good confidence in updated baseline
                        print(f"Baseline adaptively updated: change magnitude {change_magnitude:.3f}")
                        return True
                
                self.current_center_samples.clear()
        
        return False
    
    def auto_adapt_to_position_change(self, current_features):
        """Automatically adapt to position changes using existing training data"""
        if not AUTO_ADAPTATION_MODE or self.adaptation_in_progress:
            return False
        
        # Check if enough time has passed since last adaptation
        if time.time() - self.last_auto_adaptation < 10.0:  # Wait 10 seconds between adaptations
            return False
        
        if len(self.training_data['features']) < 20:
            return False  # Need enough training data
        
        try:
            self.adaptation_in_progress = True
            print("🔄 Auto-adapting to position change using existing training data...")
            
            # Calculate feature offset from current to original baseline
            if self.baseline_features is not None:
                feature_offset = current_features[:6] - self.baseline_features[:6]
                print(f"Detected position offset: {feature_offset[:4]}")
                
                # Update baseline to current position
                new_baseline = current_features.copy()
                screen_center = [SCREEN_W // 2, SCREEN_H // 2]
                
                # Temporarily update baseline for testing
                old_baseline = self.baseline_features.copy()
                self.baseline_features = new_baseline
                self.baseline_screen_pos = np.array(screen_center)
                
                # Test prediction accuracy with new baseline on recent samples
                if len(self.position_drift_buffer) > 10:
                    test_accuracy = self._test_baseline_accuracy()
                    
                    if test_accuracy > 0.7:  # Good accuracy with new baseline
                        self.baseline_confidence = 0.8
                        self.last_auto_adaptation = time.time()
                        print(f"✅ Auto-adaptation successful! New baseline accuracy: {test_accuracy:.2f}")
                        
                        # Save the updated model
                        self.save_models()
                        return True
                    else:
                        # Revert to old baseline
                        self.baseline_features = old_baseline
                        print(f"❌ Auto-adaptation failed. Accuracy too low: {test_accuracy:.2f}")
                
                return False
                
        except Exception as e:
            print(f"Error in auto-adaptation: {e}")
            return False
        finally:
            self.adaptation_in_progress = False
    
    def _test_baseline_accuracy(self):
        """Test accuracy of current baseline using recent position data"""
        try:
            if len(self.position_drift_buffer) < 5:
                return 0.0
            
            # Use recent samples to test accuracy
            recent_samples = list(self.position_drift_buffer)[-10:]
            errors = []
            
            for sample in recent_samples:
                features, expected_center = sample
                
                # Predict using current baseline
                if self.baseline_features is not None:
                    relative_feat = features.copy()
                    relative_feat[:4] = features[:4] - self.baseline_features[:4]
                    relative_feat[4:6] = features[4:6] - self.baseline_features[4:6]
                    
                    # Simple distance-based accuracy test
                    feature_stability = np.mean(np.abs(relative_feat[:4]))
                    errors.append(feature_stability)
            
            if errors:
                avg_error = np.mean(errors)
                accuracy = max(0.0, 1.0 - (avg_error / 0.2))  # Normalize to 0-1
                return accuracy
            
            return 0.0
            
        except Exception as e:
            print(f"Error testing baseline accuracy: {e}")
            return 0.0
    
    def intelligent_position_tracking(self, current_features, predicted_pos):
        """Track position changes and trigger intelligent adaptations"""
        screen_center = [SCREEN_W // 2, SCREEN_H // 2]
        
        # Store sample if near screen center for drift analysis
        distance_from_center = np.sqrt((predicted_pos[0] - screen_center[0])**2 + 
                                     (predicted_pos[1] - screen_center[1])**2)
        
        if distance_from_center < 200:  # Within 200 pixels of center
            self.position_drift_buffer.append((current_features.copy(), screen_center))
        
        # Check if auto-adaptation should be triggered
        if (self.baseline_confidence < ADAPTATION_CONFIDENCE_THRESHOLD and 
            not self.adaptation_in_progress and
            AUTO_ADAPTATION_MODE):
            
            # Collect a few stable samples before adapting
            self.adaptation_samples.append(current_features.copy())
            
            if len(self.adaptation_samples) >= 5:
                # Use averaged features for adaptation
                avg_features = np.mean(self.adaptation_samples, axis=0)
                success = self.auto_adapt_to_position_change(avg_features)
                
                if success:
                    self.adaptation_samples.clear()
                    return True
                else:
                    # Keep only recent samples
                    while len(self.adaptation_samples) > 3:
                        self.adaptation_samples.popleft()
        
        return False
    
    def add_training_data(self, features, targets, is_baseline=False, should_train=None):
        """Add new training data and decide whether to retrain"""
        initial_count = len(self.training_data['features'])
        
        # If this is the first point (center/baseline), store it specially
        if is_baseline and len(features) > 0:
            self.set_baseline(features[0], targets[0])
        
        for feat, target in zip(features, targets):
            # Create relative features if we have a baseline
            if self.baseline_features is not None:
                # Calculate relative eye angles from baseline
                relative_feat = feat.copy()
                relative_feat[:4] = feat[:4] - self.baseline_features[:4]  # relative eye positions
                relative_feat[4:6] = feat[4:6] - self.baseline_features[4:6]  # relative head pose
                self.training_data['features'].append(relative_feat)
            else:
                # No baseline yet, use absolute features
                self.training_data['features'].append(feat)
            
            self.training_data['targets_x'].append(target[0])
            self.training_data['targets_y'].append(target[1])
        
        final_count = len(self.training_data['features'])
        added_count = final_count - initial_count
        
        print(f"[DATA] Added {added_count} samples ({'baseline-relative' if self.baseline_features is not None else 'absolute'}). Total: {final_count}")
        
        # Auto-save training data periodically
        if final_count % AUTO_SAVE_FREQUENCY == 0:
            enhanced_data = {
                'training_data': self.training_data,
                'baseline_features': self.baseline_features,
                'baseline_screen_pos': self.baseline_screen_pos,
                'calibration_session_id': self.calibration_session_id
            }
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(enhanced_data, f)
            print(f"Auto-saved training data at {final_count} samples")
        
        # Decide whether to train based on strategy
        should_retrain = False
        
        if should_train is not None:
            should_retrain = should_train
        elif TRAINING_MODE == "ALWAYS":
            should_retrain = True
            print("Training mode: ALWAYS - Retraining after calibration")
        elif TRAINING_MODE == "INCREMENTAL":
            self.incremental_samples_since_training += added_count
            if self.incremental_samples_since_training >= INCREMENTAL_TRAINING_SAMPLES:
                should_retrain = True
                print(f"Training mode: INCREMENTAL - Retraining with {self.incremental_samples_since_training} new samples")
                self.incremental_samples_since_training = 0
            else:
                print(f"Training mode: INCREMENTAL - {self.incremental_samples_since_training}/{INCREMENTAL_TRAINING_SAMPLES} samples until next training")
        elif TRAINING_MODE == "THRESHOLD":
            if final_count >= RETRAIN_THRESHOLD and (final_count - added_count) < RETRAIN_THRESHOLD:
                should_retrain = True
                print(f"Training mode: THRESHOLD - Retraining at {final_count} samples")
        elif TRAINING_MODE == "MANUAL":
            print("Training mode: MANUAL - Use 's' key to train manually")
            should_retrain = False
        
        if should_retrain:
            print("[TRAINING] Model will be retrained after this calibration.")
        
        return should_retrain
    
    def train_models(self):
        """Enhanced training with better data preparation and validation"""
        if len(self.training_data['features']) < 10:
            print("Not enough training data to train models")
            return False
        
        try:
            print(f"🔄 [TRAINING] Training models with {len(self.training_data['features'])} samples...")
            
            # Prepare data
            X = np.array(self.training_data['features'])
            y_x = np.array(self.training_data['targets_x'])
            y_y = np.array(self.training_data['targets_y'])
            
            # Data validation and cleaning
            print(f"Data shape: X={X.shape}, y_x={y_x.shape}, y_y={y_y.shape}")
            
            # Remove any invalid samples
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_x) | np.isnan(y_y))
            if not valid_mask.all():
                print(f"Removing {(~valid_mask).sum()} invalid samples")
                X = X[valid_mask]
                y_x = y_x[valid_mask]
                y_y = y_y[valid_mask]
            
            if len(X) < 10:
                print("Not enough valid samples after cleaning")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            print(f"Feature scaling completed. Range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
            
            # Create and train models with improved parameters
            print("Creating enhanced neural network models...")
            self.model_x = self.create_model()
            self.model_y = self.create_model()
            
            print("Training X-coordinate model...")
            self.model_x.fit(X_scaled, y_x)
            
            print("Training Y-coordinate model...")
            self.model_y.fit(X_scaled, y_y)
            
            # Calculate comprehensive accuracy metrics
            pred_x = self.model_x.predict(X_scaled)
            pred_y = self.model_y.predict(X_scaled)
            
            mse_x = mean_squared_error(y_x, pred_x)
            mse_y = mean_squared_error(y_y, pred_y)
            
            # Calculate pixel-level accuracy
            pixel_errors = np.sqrt((pred_x - y_x)**2 + (pred_y - y_y)**2)
            mean_pixel_error = np.mean(pixel_errors)
            std_pixel_error = np.std(pixel_errors)
            
            # Accuracy percentage (within 50 pixels)
            accuracy_50px = np.mean(pixel_errors < 50) * 100
            accuracy_100px = np.mean(pixel_errors < 100) * 100
            
            print(f"📊 Training Results:")
            print(f"   MSE - X: {mse_x:.1f}, Y: {mse_y:.1f}")
            print(f"   Mean pixel error: {mean_pixel_error:.1f} ± {std_pixel_error:.1f} pixels")
            print(f"   Accuracy within 50px: {accuracy_50px:.1f}%")
            print(f"   Accuracy within 100px: {accuracy_100px:.1f}%")
            
            # Track training history
            training_result = {
                'sample_count': len(X),
                'mse_x': mse_x,
                'mse_y': mse_y,
                'mean_pixel_error': mean_pixel_error,
                'accuracy_50px': accuracy_50px,
                'accuracy_100px': accuracy_100px,
                'timestamp': time.time()
            }
            self.training_history.append(training_result)
            
            # Keep only last 10 training results
            if len(self.training_history) > 10:
                self.training_history = self.training_history[-10:]
            
            self.last_training_sample_count = len(X)
            
            # Save models
            self.save_models()
            
            print(f"✅ [TRAINING] Model training completed successfully!")
            print(f"   [TRAINING] Latest accuracy within 50px: {accuracy_50px:.1f}%")
            print(f"   [TRAINING] Latest mean pixel error: {mean_pixel_error:.1f}px")
            self.print_model_status()
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_realtime_sample(self, features, cursor_pos):
        """Add a sample during real-time usage for continuous learning"""
        # Only add samples when user appears to be looking at cursor position
        # This helps with continuous model improvement
        
        # Predict where we think they're looking
        predicted_pos = self.predict(features)
        
        # If prediction is reasonably close to actual cursor position, add as training sample
        distance = np.sqrt((predicted_pos[0] - cursor_pos[0])**2 + (predicted_pos[1] - cursor_pos[1])**2)
        
        if distance < 100:  # Within 100 pixels - likely accurate
            print("[LEARNING] Adding new realtime sample for continuous learning.")
            # Convert to relative features if we have baseline
            if self.baseline_features is not None:
                relative_feat = features.copy()
                relative_feat[:4] = features[:4] - self.baseline_features[:4]
                relative_feat[4:6] = features[4:6] - self.baseline_features[4:6]
                self.training_data['features'].append(relative_feat)
            else:
                self.training_data['features'].append(features)
            
            self.training_data['targets_x'].append(cursor_pos[0])
            self.training_data['targets_y'].append(cursor_pos[1])
            
            self.incremental_samples_since_training += 1
            
            # Check if we should retrain
            if (TRAINING_MODE == "INCREMENTAL" and 
                self.incremental_samples_since_training >= INCREMENTAL_TRAINING_SAMPLES):
                print(f"🔄 [LEARNING] Continuous learning: retraining with {self.incremental_samples_since_training} new samples")
                self.train_models()
                self.incremental_samples_since_training = 0
                return True
        
        return False
    
    def predict(self, features):
        """Predict screen coordinates from gaze features with enhanced accuracy and adaptation"""
        if self.model_x is None or self.model_y is None:
            return self.fallback_predict(features)
        
        try:
            # More frequent position change checks for better responsiveness
            current_time = time.time()
            if current_time - self.last_position_check > 0.5:  # Check every 0.5 seconds for faster adaptation
                self.last_position_check = current_time
                if self.baseline_features is not None:
                    position_changed, reason = self.check_position_change(features)
                    if position_changed:
                        if not self.adaptation_in_progress:
                            print(f"🔄 Position change detected: {reason}")
                            print(f"Baseline confidence: {self.baseline_confidence:.2f}")
                            
                            # Enhanced intelligent auto-adaptation with faster response
                            if AUTO_ADAPTATION_MODE and self.baseline_confidence < ADAPTATION_CONFIDENCE_THRESHOLD:
                                print("🤖 Triggering enhanced intelligent auto-adaptation...")
            
            # Convert to relative features with improved normalization
            if self.baseline_features is not None:
                relative_feat = features.copy()
                # Enhanced relative feature calculation with weighted normalization
                # Eye positions (more weight to vertical component for better top area accuracy)
                relative_feat[0] = features[0] - self.baseline_features[0]  # Left eye X
                relative_feat[1] = (features[1] - self.baseline_features[1]) * 1.5  # Left eye Y (significantly enhanced)
                relative_feat[2] = features[2] - self.baseline_features[2]  # Right eye X
                relative_feat[3] = (features[3] - self.baseline_features[3]) * 1.5  # Right eye Y (significantly enhanced)
                
                # Head pose with enhanced sensitivity - giving more weight to head movements
                relative_feat[4] = (features[4] - self.baseline_features[4]) * 1.8  # Yaw (significantly enhanced)
                relative_feat[5] = (features[5] - self.baseline_features[5]) * 2.0  # Pitch (significantly enhanced for vertical)
                
                features_to_use = relative_feat
            else:
                features_to_use = features
            
            # Apply feature scaling with outlier protection
            features_scaled = self.scaler.transform([features_to_use])
            
            # Get raw predictions
            x_pred = self.model_x.predict(features_scaled)[0]
            y_pred = self.model_y.predict(features_scaled)[0]
            
            # Enhanced non-linear correction for top area accuracy
            # Apply stronger correction to Y coordinate for better top area tracking
            if y_pred < SCREEN_H * 0.3:  # Top 30% of screen
                # Apply non-linear correction to improve accuracy in top area
                y_correction = (SCREEN_H * 0.3 - y_pred) / (SCREEN_H * 0.3) * 20  # Max 20px correction
                y_pred = max(0, y_pred - y_correction)  # Shift upward slightly
            
            # Apply confidence-based smoothing with improved weighting
            confidence_factor = max(0.6, self.baseline_confidence)
            
            # Ensure predictions stay within screen bounds
            x_pred = np.clip(x_pred, 0, SCREEN_W - 1)
            y_pred = np.clip(y_pred, 0, SCREEN_H - 1)
            
            predicted_pos = (x_pred, y_pred)
            
            # Intelligent position tracking and auto-adaptation
            adaptation_triggered = self.intelligent_position_tracking(features, predicted_pos)
            if adaptation_triggered:
                print("✅ Intelligent auto-adaptation completed!")
            
            # Try adaptive baseline update (legacy method)
            self.update_baseline_adaptively(features, predicted_pos)
            
            return float(x_pred), float(y_pred)
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.fallback_predict(features)
    
    def fallback_predict(self, features):
        """Simple fallback prediction method"""
        # Map normalized iris positions directly to screen coordinates
        lx, ly, rx, ry = features[:4]
        avg_x = (lx + rx) / 2
        avg_y = (ly + ry) / 2
        
        screen_x = np.clip(avg_x * SCREEN_W, 0, SCREEN_W - 1)
        screen_y = np.clip(avg_y * SCREEN_H, 0, SCREEN_H - 1)
        
        return float(screen_x), float(screen_y)

def quick_center_recalibration(cap):
    """Quick recalibration of just the center point when position changes detected"""
    win_name = "QUICK RECALIBRATION"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    if USE_FULLSCREEN_CALIB_UI:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("Quick center recalibration...")
    
    sample_buf = deque(maxlen=STABILITY_WINDOW)
    collected_samples = 0
    target_samples = 10
    
    start_time = time.time()
    
    while collected_samples < target_samples:
        if time.time() - start_time > 15:  # Timeout after 15 seconds
            break
            
        ret, frame = cap.read()
        if not ret:
            continue
        fh, fw = frame.shape[:2]
        frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_mesh.process(frgb)

        # UI
        ui = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        ui[:] = (25, 25, 25)
        
        # Center point
        cx, cy = SCREEN_W // 2, SCREEN_H // 2
        show_calib_point(ui, cx, cy, r=25, color=(0, 255, 255))
        
        draw_text(ui, "QUICK RECALIBRATION", (40, 60), 1.0, 3, (0, 255, 255))
        draw_text(ui, "Look at the CENTER point to update your baseline", (40, 120), 0.8, 2, (255, 255, 255))
        draw_text(ui, f"Samples: {collected_samples}/{target_samples}", (40, 160), 0.7, 2, (255, 240, 180))
        draw_text(ui, "This will only take a few seconds...", (40, 200), 0.6, 2, (220, 220, 255))
        
        cv2.imshow(win_name, ui)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break

        if out.multi_face_landmarks:
            lms = out.multi_face_landmarks[0].landmark
            
            # Extract features (same as main calibration)
            li = iris_center(lms, L_IRIS, fw, fh)
            ri = iris_center(lms, R_IRIS, fw, fh)
            lnorm = eye_box_norm(lms, L_OUT, L_IN, fw, fh, li)
            rnorm = eye_box_norm(lms, R_OUT, R_IN, fw, fh, ri)
            yaw, pitch = head_pose_yaw_pitch(lms, fw, fh, fw, fh)

            # Simplified 6-dimensional feature vector
            feat = np.array([
                lnorm[0], lnorm[1], rnorm[0], rnorm[1], # 4 eye features
                yaw / 45.0,  # Normalized yaw
                pitch / 45.0 # Normalized pitch
            ], dtype=np.float64)
            
            sample_buf.append(feat)
            
            if len(sample_buf) >= STABILITY_WINDOW:
                stds = np.std(np.stack(sample_buf, axis=0), axis=0)
                if np.all(stds[:4] < STABILITY_STD_MAX):
                    avg_feat = np.mean(np.stack(sample_buf, axis=0), axis=0)
                    collected_samples += 1
                    sample_buf.clear()
                    time.sleep(0.1)
    
    cv2.destroyWindow(win_name)
    
    if collected_samples >= 5:  # At least 5 good samples
        # Update baseline with averaged features
        final_features = np.mean([sample_buf[i] if i < len(sample_buf) else avg_feat 
                                for i in range(min(collected_samples, len(sample_buf)))], axis=0)
        gaze_model.set_baseline(final_features, [SCREEN_W // 2, SCREEN_H // 2])
        print(f"✓ Baseline updated from {collected_samples} samples")
        return True
    else:
        print("❌ Quick recalibration failed - not enough stable samples")
        return False

# Global model instance
gaze_model = GazeModel()

def draw_text(img, txt, org, scale=0.6, thick=2, color=(255,255,255)):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def show_calib_point(img, x, y, r=14, color=(0,255,0)):
    cv2.circle(img, (int(x), int(y)), r, color, -1, lineType=cv2.LINE_AA)

# =========================
# CALIBRATION
# =========================
def run_calibration(cap):
    # Create calibration UI window
    win_name = "ENHANCED CALIBRATION"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    if USE_FULLSCREEN_CALIB_UI:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    features = []  # list of [lx, ly, rx, ry, yaw, pitch, ...]
    targets = []   # list of [screen_x, screen_y]
    baseline_collected = False

    # Use a reduced set of calibration points for faster calibration
    reduced_calib_points = [
        # Center point (baseline)
        (0.5, 0.5),
        # Corners - essential for mapping full range
        (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
        # Top area - important for accuracy
        (0.5, 0.1), (0.3, 0.1), (0.7, 0.1),
        # Middle points
        (0.3, 0.5), (0.7, 0.5),
        # Bottom area
        (0.5, 0.9), (0.3, 0.9), (0.7, 0.9),
        # Sides
        (0.1, 0.5), (0.9, 0.5)
    ]
    
    point_idx = 0
    sample_buf = deque(maxlen=STABILITY_WINDOW)
    collected_for_point = 0
    last_status = ""
    
    # Increment calibration session
    gaze_model.calibration_session_id += 1

    print(f"Starting FASTER calibration with {len(reduced_calib_points)} points...")
    print("PHASE 1: Establishing baseline from center point")

    while point_idx < len(reduced_calib_points):
        # read a camera frame
        ret, frame = cap.read()
        if not ret:
            continue
        fh, fw = frame.shape[:2]
        frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_mesh.process(frgb)

        # prepare UI canvas
        ui = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        ui[:] = (25, 25, 25)

        # current target on screen
        nx, ny = reduced_calib_points[point_idx]
        tx, ty = int(nx * SCREEN_W), int(ny * SCREEN_H)
        
        # Determine if this is the center/baseline point or top area point
        is_center_point = (point_idx == 0)
        is_top_area_point = (ny < 0.3)  # Top 30% of screen
        
        # Faster calibration with fewer samples per point
        if is_center_point:
            samples_needed = BASELINE_SAMPLES
            stability_threshold = STABILITY_STD_MAX * 1.2  # More relaxed for faster collection
        elif is_top_area_point:
            samples_needed = TOP_AREA_SAMPLES  # Extra samples for top area
            stability_threshold = TOP_AREA_STABILITY_MAX * 1.2  # More relaxed for faster collection
        else:
            samples_needed = SAMPLES_PER_POINT
            stability_threshold = STABILITY_STD_MAX * 1.2  # More relaxed for faster collection

        # draw points with enhanced visualization
        for i, (gx, gy) in enumerate(reduced_calib_points):
            px, py = int(gx*SCREEN_W), int(gy*SCREEN_H)
            if i < point_idx:
                c = (60, 140, 60)  # completed - green
                r = 8
            elif i == point_idx:
                if is_center_point:
                    c = (0, 255, 255)  # center point - cyan
                    r = 20
                else:
                    c = (40, 180, 250)  # current - yellow  
                    r = 15
            else:
                c = (80, 80, 80)  # future - gray
                r = 4
            show_calib_point(ui, px, py, r=r, color=c)

        # Enhanced progress indicators
        progress_pct = (point_idx / len(reduced_calib_points)) * 100
        
        if is_center_point:
            draw_text(ui, "PHASE 1: BASELINE CALIBRATION", (40, 60), 1.0, 3, (0, 255, 255))
            draw_text(ui, "Look directly at the CENTER point - This establishes your natural eye position", (40, 100), 0.8, 2, (220,255,220))
            draw_text(ui, "Keep your head straight and look naturally at the center", (40, 130), 0.7, 2, (255,255,255))
        elif is_top_area_point:
            draw_text(ui, "PHASE 2: TOP AREA ENHANCEMENT", (40, 60), 0.9, 2, (255, 255, 0))
            draw_text(ui, "Look at the highlighted point - EXTRA SAMPLES for better top accuracy", (40, 100), 0.8, 2, (255,255,120))
            draw_text(ui, "Keep head steady, move only EYES upward", (40, 130), 0.7, 2, (255,220,120))
        else:
            phase = "PHASE 2: GENERAL SCREEN MAPPING"
            draw_text(ui, phase, (40, 60), 0.9, 2, (200,255,200))
            draw_text(ui, "Look at the highlighted point - Move only your EYES, keep head steady", (40, 100), 0.8, 2, (220,220,255))
        
        draw_text(ui, f"Point {point_idx+1}/{len(reduced_calib_points)} ({progress_pct:.1f}%)  Samples: {collected_for_point}/{samples_needed}", (40, 160), 0.7, 2, (255,240,180))
        
        if last_status:
            color = (120, 255, 120) if "Captured" in last_status else (255, 210, 120)
            draw_text(ui, last_status, (40, 190), 0.7, 2, color)
        
        # Show session info
        draw_text(ui, f"Session #{gaze_model.calibration_session_id} | Training samples: {len(gaze_model.training_data['features'])}", (40, 220), 0.6, 2, (180, 180, 255))
        if baseline_collected:
            draw_text(ui, "✓ Baseline established - Now mapping full screen", (40, 240), 0.6, 2, (120, 255, 120))

        cv2.imshow(win_name, ui)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break
        elif key == ord('s'):  # Skip current point
            point_idx += 1
            collected_for_point = 0
            sample_buf.clear()
            continue

        if out.multi_face_landmarks:
            lms = out.multi_face_landmarks[0].landmark

            # iris centers
            li = iris_center(lms, L_IRIS, fw, fh)
            ri = iris_center(lms, R_IRIS, fw, fh)

            # normalized within eye boxes
            lnorm = eye_box_norm(lms, L_OUT, L_IN, fw, fh, li)  # [ex, ez]
            rnorm = eye_box_norm(lms, R_OUT, R_IN, fw, fh, ri)

            # head pose
            yaw, pitch = head_pose_yaw_pitch(lms, fw, fh, fw, fh)

            # Simplified 6-dimensional feature vector
            feat = np.array([
                lnorm[0], lnorm[1], rnorm[0], rnorm[1], # 4 eye features
                yaw / 45.0,  # Normalized yaw
                pitch / 45.0 # Normalized pitch
            ], dtype=np.float64)
            
            sample_buf.append(feat)

            # stability check on rolling window with appropriate threshold
            if len(sample_buf) >= STABILITY_WINDOW:
                stds = np.std(np.stack(sample_buf, axis=0), axis=0)
                
                # Check for both eye and head stability
                eye_stable = np.all(stds[:4] < stability_threshold)
                head_stable = np.all(stds[4:6] < 0.02) # Corresponds to ~1 degree change

                if eye_stable and head_stable:
                    # Accept averaged sample
                    avg_feat = np.mean(np.stack(sample_buf, axis=0), axis=0)
                    features.append(avg_feat)
                    targets.append([tx, ty])
                    collected_for_point += 1
                    
                    # Enhanced status messages
                    status_suffix = " (BASELINE)" if is_center_point else " (TOP AREA)" if is_top_area_point else ""
                    last_status = f"Captured! ({collected_for_point}/{samples_needed}){status_suffix}"
                    sample_buf.clear()
                    time.sleep(0.05)
                else:
                    stability_type = "baseline stable" if is_center_point else "top area stable" if is_top_area_point else "eyes stable"
                    last_status = f"Hold steady... ({stability_type})"
                    # Remove oldest sample to keep rolling
                    if len(sample_buf) >= STABILITY_WINDOW:
                        sample_buf.popleft()
            else:
                last_status = f"Detecting gaze... ({len(sample_buf)}/{STABILITY_WINDOW})"

            if collected_for_point >= samples_needed:
                # Mark baseline as collected if this was the center point
                if is_center_point:
                    baseline_collected = True
                    print("✓ Baseline established from center point")
                elif is_top_area_point:
                    print(f"✓ Top area point {point_idx} enhanced with {collected_for_point} samples")
                
                collected_for_point = 0
                point_idx += 1
                sample_buf.clear()
                last_status = "Moving to next point..."
                time.sleep(0.15)

    cv2.destroyWindow(win_name)

    print(f"Calibration completed with {len(features)} samples")

    if len(features) < 5:
        print("Warning: Very few calibration samples collected!")
        if len(features) == 0:
            print("No samples collected, using default training data")
            return False

    # Add calibration data to the model's training dataset
    print(f"Adding {len(features)} new calibration samples to training data...")
    # Pass baseline flag for the first sample (center point)
    should_retrain = gaze_model.add_training_data(features, targets, is_baseline=True)
    
    # Train the model based on strategy
    if should_retrain:
        success = gaze_model.train_models()
        if not success and len(gaze_model.training_data['features']) == 0:
            print("Model training failed and no existing data, using fallback prediction")
            return False
    else:
        print("Skipping training based on current strategy")
    
    return True

# =========================
# CONTROL LOOP
# =========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    # Prepare initial calibration
    print("Starting enhanced gaze tracking with machine learning...")
    print(f"Existing training samples: {len(gaze_model.training_data['features'])}")
    
    # Always run calibration to collect more training data
    calib_success = run_calibration(cap)
    if not calib_success:
        print("Calibration failed!")
        return
    else:
        print("[INFO] Calibration completed successfully.")
    print_model_status()
    print("Control mode starting.")
    print(f"Training mode: {TRAINING_MODE}")
    print(f"Auto-adaptation: {'ENABLED' if AUTO_ADAPTATION_MODE else 'DISABLED'}")
    print("Commands: 'c'=full recalib, 'q'=quick center recalib, 'v'=overlay, 's'=save, 't'=train, ESC=quit")

    # Smoothing buffers
    sx_buf = deque(maxlen=SMOOTHING_WINDOW)
    sy_buf = deque(maxlen=SMOOTHING_WINDOW)

    # Blink/wink state
    blink_frames = 0
    wink_right_frames = 0
    last_left_click_time = 0
    last_right_click_time = 0
    overlay = SHOW_DEBUG_OVERLAY
    
    # Performance tracking
    frame_count = 0
    accuracy_samples = deque(maxlen=100)
    last_cursor_pos = (0, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        fh, fw = frame.shape[:2]
        frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_mesh.process(frgb)
        
        frame_count += 1

        if out.multi_face_landmarks:
            lms = out.multi_face_landmarks[0].landmark

            li = iris_center(lms, L_IRIS, fw, fh)
            ri = iris_center(lms, R_IRIS, fw, fh)

            lnorm = eye_box_norm(lms, L_OUT, L_IN, fw, fh, li)
            rnorm = eye_box_norm(lms, R_OUT, R_IN, fw, fh, ri)

            yaw, pitch = head_pose_yaw_pitch(lms, fw, fh, fw, fh)

            # Simplified 6-dimensional feature vector
            feats = np.array([
                lnorm[0], lnorm[1], rnorm[0], rnorm[1], # 4 eye features
                yaw / 45.0,  # Normalized yaw
                pitch / 45.0 # Normalized pitch
            ], dtype=np.float64)

            # predict screen coordinates using ML model
            sx, sy = gaze_model.predict(feats)

            # Enhanced adaptive smoothing for better stability
            sx_buf.append(sx)
            sy_buf.append(sy)
            
            # Apply weighted smoothing with more weight to recent positions
            # This provides better responsiveness while maintaining stability
            if len(sx_buf) >= SMOOTHING_WINDOW:
                weights = np.linspace(0.5, 1.0, len(sx_buf))  # Increasing weights for newer samples
                weights = weights / np.sum(weights)  # Normalize weights
                
                # Apply weighted average
                msx = int(np.sum(np.array(sx_buf) * weights))
                msy = int(np.sum(np.array(sy_buf) * weights))
            else:
                # Simple average for initial frames
                msx = int(np.mean(sx_buf))
                msy = int(np.mean(sy_buf))
                
            # Apply additional non-linear smoothing for small movements (reduces jitter)
            if len(sx_buf) > 1:
                last_x, last_y = last_cursor_pos
                dist = np.sqrt((msx - last_x)**2 + (msy - last_y)**2)
                
                if dist < 10:  # Small movement - apply stronger smoothing
                    msx = int(0.7 * last_x + 0.3 * msx)
                    msy = int(0.7 * last_y + 0.3 * msy)

            # move cursor
            pyautogui.moveTo(msx, msy)
            
            # Continuous learning - add samples when prediction is accurate
            if frame_count % 30 == 0:  # Every 30 frames (about once per second)
                current_cursor = pyautogui.position()
                learning_success = gaze_model.add_realtime_sample(feats, current_cursor)
                if learning_success:
                    print("🧠 [LEARNING] Model improved through continuous learning!")
            
            # Track cursor movement for performance analysis
            current_cursor_pos = (msx, msy)
            if frame_count > 1:
                cursor_movement = np.sqrt((current_cursor_pos[0] - last_cursor_pos[0])**2 + 
                                        (current_cursor_pos[1] - last_cursor_pos[1])**2)
                accuracy_samples.append(cursor_movement)
            last_cursor_pos = current_cursor_pos

            # Blink / wink detection (improved thresholds)
            l_ear = ear_ratio(lms, L_UP, L_DN, L_OUT, L_IN, fw, fh)
            r_ear = ear_ratio(lms, R_UP, R_DN, R_OUT, R_IN, fw, fh)

            # Long blink for left click
            if l_ear < BLINK_EAR_THRESH and r_ear < BLINK_EAR_THRESH:
                blink_frames += 1
            else:
                if blink_frames >= BLINK_MIN_FRAMES and (time.time() - last_left_click_time > 0.6):
                    pyautogui.click(button='left')
                    last_left_click_time = time.time()
                    print(f"Left click at ({msx}, {msy})")
                blink_frames = 0

            # Right wink for right click (right eye much more closed than left)
            if (r_ear + 1e-6) < (l_ear - WINK_RIGHT_DIFF):
                wink_right_frames += 1
            else:
                if wink_right_frames >= WINK_MIN_FRAMES and (time.time() - last_right_click_time > 1.0):
                    pyautogui.click(button='right')
                    last_right_click_time = time.time()
                    print(f"Right click at ({msx}, {msy})")
                wink_right_frames = 0

            if overlay:
                # Enhanced visual overlay
                cv2.circle(frame, tuple(np.int32(li)), 4, (0,255,0), -1)
                cv2.circle(frame, tuple(np.int32(ri)), 4, (0,255,0), -1)
                
                # Feature display
                draw_text(frame, f"L-norm: {lnorm[0]:.2f},{lnorm[1]:.2f}", (10, 30), 0.5, 2, (255,255,255))
                draw_text(frame, f"R-norm: {rnorm[0]:.2f},{rnorm[1]:.2f}", (10, 50), 0.5, 2, (255,255,255))
                draw_text(frame, f"Head: {yaw:.1f}°/{pitch:.1f}°", (10, 70), 0.5, 2, (255,220,220))
                draw_text(frame, f"EAR L/R: {l_ear:.2f}/{r_ear:.2f}", (10, 90), 0.5, 2, (220,255,220))
                draw_text(frame, f"Cursor: {msx},{msy}", (10, 110), 0.5, 2, (220,220,255))
                
                # Model info with training history
                model_type = "Neural Net" if USE_NEURAL_NETWORK else "Random Forest"
                training_samples = len(gaze_model.training_data['features'])
                draw_text(frame, f"Model: {model_type} ({training_samples} samples)", (10, 130), 0.5, 2, (180,255,180))
                draw_text(frame, f"Training: {TRAINING_MODE}", (10, 150), 0.5, 2, (180,180,255))
                
                # Show latest training results
                if len(gaze_model.training_history) > 0:
                    latest = gaze_model.training_history[-1]
                    draw_text(frame, f"Accuracy: {latest['accuracy_50px']:.1f}% (50px)", (10, 170), 0.5, 2, (120,255,120))
                    draw_text(frame, f"Avg Error: {latest['mean_pixel_error']:.1f}px", (10, 190), 0.5, 2, (255,255,120))
                
                # Incremental training progress
                if TRAINING_MODE == "INCREMENTAL":
                    progress = gaze_model.incremental_samples_since_training
                    draw_text(frame, f"Next training: {progress}/{INCREMENTAL_TRAINING_SAMPLES}", (10, 210), 0.5, 2, (255,180,180))
                
                # Performance metrics
                if len(accuracy_samples) > 10:
                    avg_movement = np.mean(accuracy_samples)
                    draw_text(frame, f"Cursor Movement: {avg_movement:.1f}px", (10, 230), 0.5, 2, (255,180,180))
                
                # Position stability and auto-adaptation status  
                confidence_color = (120, 255, 120) if gaze_model.baseline_confidence > 0.7 else (255, 255, 120) if gaze_model.baseline_confidence > 0.4 else (255, 120, 120)
                draw_text(frame, f"Position Confidence: {gaze_model.baseline_confidence:.2f}", (10, 250), 0.5, 2, confidence_color)
                
                adaptation_status = "ADAPTING..." if gaze_model.adaptation_in_progress else f"AUTO-ADAPT: {'ON' if AUTO_ADAPTATION_MODE else 'OFF'}"
                adaptation_color = (255, 255, 120) if gaze_model.adaptation_in_progress else (120, 255, 120) if AUTO_ADAPTATION_MODE else (120, 120, 120)
                draw_text(frame, adaptation_status, (10, 270), 0.5, 2, adaptation_color)
                
                # Instructions
                draw_text(frame, "Blink=Left | Wink=Right | 'c'=FullRecalib | 'q'=QuickRecalib | 'v'=Overlay | ESC=Quit", 
                         (10, fh-15), 0.42, 2, (255,240,180))
        else:
            # no face detected
            if overlay:
                draw_text(frame, "No face detected", (10, 30), 0.6, 2, (0,0,255))

        cv2.imshow("Enhanced Gaze Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break
        elif key == RECALIB_KEY:
            print("Full recalibration...")
            calib_success = run_calibration(cap)
            if calib_success:
                sx_buf.clear(); sy_buf.clear()
                print("Full recalibration complete.")
            else:
                print("Full recalibration failed!")
        elif key == ord('q'):
            print("Quick center recalibration...")
            if quick_center_recalibration(cap):
                sx_buf.clear(); sy_buf.clear()
                print("Quick recalibration complete.")
            else:
                print("Quick recalibration failed!")
        elif key == ord('v'):
            overlay = not overlay
            print(f"Overlay {'enabled' if overlay else 'disabled'}")
        elif key == ord('s'):
            print("Saving model...")
            gaze_model.save_models()
        elif key == ord('t'):
            if len(gaze_model.training_data['features']) >= 10:
                print("Manual training initiated...")
                gaze_model.train_models()
            else:
                print("Not enough training data for manual training")

    cap.release()
    cv2.destroyAllWindows()
    
    # Final save before exit
    print("Saving final model state...")
    gaze_model.save_models()
    print(f"Session complete. Total training samples: {len(gaze_model.training_data['features'])}")

def print_model_status():
    print("\n========== MODEL STATUS ==========")
    print(f"Model X trained: {'YES' if gaze_model.model_x is not None else 'NO'}")
    print(f"Model Y trained: {'YES' if gaze_model.model_y is not None else 'NO'}")
    print(f"Training samples: {len(gaze_model.training_data['features'])}")
    print(f"Feature dimension: {gaze_model.feature_dim}")
    if gaze_model.training_history:
        latest = gaze_model.training_history[-1]
        print(f"Latest accuracy within 50px: {latest['accuracy_50px']:.1f}%")
        print(f"Latest mean pixel error: {latest['mean_pixel_error']:.1f}px")
    else:
        print("No training history available.")
    print("==================================\n")

if __name__ == "__main__":
    main()
