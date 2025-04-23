import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # For Linux
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # For Windows

import cv2 
import numpy as np
import mtcnn
from architecture import *
import sys
import time
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine
import argparse
from collections import OrderedDict, deque
import dlib
from imutils import face_utils
import mediapipe as mp
import logging
from datetime import datetime

# Add this helper function after the imports

def write_result_to_file(result_file, message):
    """Write authentication result to a file if specified"""
    if result_file:
        try:
            with open(result_file, 'w') as f:
                f.write(message)
            print(f"Result written to: {result_file}")
        except Exception as e:
            print(f"Error writing result to file: {e}")
            # Try to write a simpler error message
            try:
                with open(result_file, 'w') as f:
                    f.write(f"Authentication failed\nError: {str(e)}")
            except:
                pass

class FaceRecognizer:
    """Responsible for face detection, recognition and authentication"""
    
    def __init__(self, display_ui=True):
        """
        Initialize the Face Recognition system
        
        Args:
            display_ui: Whether to display the UI
        """
        # Set up logging and suppress MediaPipe logging
        logging.getLogger("mediapipe").setLevel(logging.ERROR)
        os.environ["GLOG_minloglevel"] = "2"  # Suppress MediaPipe C++ logging
        
        # Default configuration
        self.config = {
            'encodings': 'encodings/encodings.pkl',
            'model': 'facenet_keras_weights.h5',
            'confidence': 0.98,
            'recognition_threshold': 0.4,
            'source': 0,
            'display_size': '1280x720',
            'blur_background': False,
            'enable_anti_spoofing': True  # Default to enabled
        }
                    
        # Set display option
        self.display_ui = display_ui
        
        # Parse display size
        self.width, self.height = map(int, self.config['display_size'].split('x'))
        
        # Initialize state variables
        self.face_encoder = None
        self.face_detector = None
        self.encoding_dict = {}
        self.face_mesh = None
        self.cap = None
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.process_times = deque(maxlen=30)
        self.start_time = 0
        
        # Face tracking
        self.face_count = 0
        self.recognized_faces = []
        self.active_auth_user = None
        self.auth_confidence = 0
        
        # Constants for face recognition
        self.confidence_t = self.config['confidence']
        self.recognition_t = self.config['recognition_threshold']
        self.required_size = (160, 160)
        
        # Configure GPU memory growth
        self._configure_gpu()
        
        # Initialize L2 normalizer
        from sklearn.preprocessing import Normalizer
        self.l2_normalizer = Normalizer('l2')
        
        # Authentication state variables
        self.auth_result = None
        self.auth_start_time = None
        self.auth_duration = 0
        self.auth_required_time = 3.0  # Need 3 seconds of continuous recognition
        self.auth_session_timeout = 30.0  # Session timeout
        
        # Blink detection if anti-spoofing is enabled
        self._init_blink_detection()
    
    def _configure_gpu(self):
        """Configure GPU memory growth to prevent TensorFlow from using all memory"""
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"GPU detected: {physical_devices[0].name}")
        else:
            print("No GPU detected, using CPU")
    
    def _init_blink_detection(self):
        """Initialize all components needed for blink detection (anti-spoofing)"""
        # Initialize MediaPipe Face Mesh for eye tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        
        # MediaPipe eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Initialize with silent output to suppress MediaPipe logging
        try:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout, sys.stderr = devnull, devnull
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Restore stdout and stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()
            print("MediaPipe Face Mesh initialized successfully")
            
        except Exception as e:
            # Ensure we restore stdout/stderr even on error
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
            if 'old_stderr' in locals():
                sys.stderr = old_stderr
            if 'devnull' in locals() and not devnull.closed:
                devnull.close()
            print(f"Error initializing MediaPipe Face Mesh: {e}")
            self.face_mesh = None
        
        # Blink detection parameters
        self.EAR_THRESHOLD_RATIO = 0.75
        self.MIN_BLINK_FRAMES = 2
        self.BLINK_COOLDOWN = 0.2
        
        # State variables for improved blink detection
        self.frames_below_threshold = 0
        self.blink_in_progress = False
        self.cooldown_active = False
        self.last_blink_time = time.time()
        self.BLINK_COUNTER = 0
        self.ear_history = deque(maxlen=60)
        self.baseline_ear = None
        self.calibration_complete = False
        self.calibration_frames = 0
        
        # Load facial landmark predictor for dlib (alternative to MediaPipe)
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            print("Dlib predictor loaded for additional anti-spoofing capability")
        else:
            print(f"Warning: {predictor_path} not found for dlib facial landmark detection")
            print(f"Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"Extract it to {predictor_path}")
    
    def reset_state(self):
        """Reset all internal tracking variables for a new session"""
        # Reset blink detection
        self.frames_below_threshold = 0
        self.blink_in_progress = False
        self.cooldown_active = False
        self.last_blink_time = time.time()
        self.BLINK_COUNTER = 0
        self.ear_history.clear()
        self.baseline_ear = None
        self.calibration_complete = False
        self.calibration_frames = 0
        
        # Reset face tracking
        self.recognized_faces = []
        self.active_auth_user = None
        self.auth_confidence = 0
        
        # Reset authentication state
        self.auth_result = None
        self.auth_start_time = None
        self.auth_duration = 0
        
        # Reset FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.process_times.clear()
        self.start_time = time.time()
        
        print("Face recognizer state reset")
    
    def load_models(self):
        """Load the face detection and recognition models"""
        print("Loading face recognition model...")
        self.face_encoder = InceptionResNetV2()
        self.face_encoder.load_weights(self.config['model'])
        
        print(f"Loading encodings from {self.config['encodings']}...")
        self.encoding_dict = self.load_pickle(self.config['encodings'])
        print(f"Loaded {len(self.encoding_dict)} face encodings")
        
        print("Initializing face detector...")
        # Remove the 'device' parameter
        self.face_detector = mtcnn.MTCNN()
        print("Face detector initialized")
        
        # Load has completed
        return True
    
    def open_video_source(self):
        """Open the video source (camera or file)"""
        source = self.config['source']
        print(f"Opening video source: {source}")
        
        try:
            if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                self.cap = cv2.VideoCapture(int(source))
            else:
                self.cap = cv2.VideoCapture(source)
                
            if not self.cap.isOpened():
                raise Exception(f"Could not open video source: {source}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Camera opened successfully
            print(f"Video source opened: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"Error opening video source: {e}")
            print("Please check your camera connection or video file path.")
            return False
    
    def normalize(self, img):
        """Normalize image for consistent processing"""
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def get_face(self, img, box):
        """Extract face from image using bounding box"""
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)

    def get_encode(self, face):
        """Get the face encoding for a detected face"""
        face = self.normalize(face)
        face = cv2.resize(face, self.required_size)
        encode = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
        return encode

    def load_pickle(self, path):
        """Load encodings from pickle file"""
        try:
            with open(path, 'rb') as f:
                encoding_dict = pickle.load(f)
            return encoding_dict
        except Exception as e:
            print(f"Error loading encodings from {path}: {e}")
            return {}
    
    def eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate the eye aspect ratio for blink detection"""
        # Get the eye landmark coordinates
        points = [landmarks[idx] for idx in eye_indices]
        
        # Calculate the vertical distances
        A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        
        # Calculate the horizontal distance
        C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def detect_faces(self, frame):
        """
        Detect and recognize faces in a frame
        
        Args:
            frame: Input frame from camera or video file
            
        Returns:
            processed_frame: Frame with detection visualization
            face_results: List of dictionaries with detection results
        """
        # For performance measurement
        process_start_time = time.time()
        
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use MTCNN to detect faces
        results = self.face_detector.detect_faces(img_rgb)
        
        # Create result lists
        face_results = []
        self.recognized_faces = []
        
        # Early return if no faces
        if not results:
            print("No faces detected in frame")
            fps_text = f"FPS: {int(self.fps)}" if self.fps > 0 else "FPS: calculating..."
            result_img = frame.copy()
            cv2.putText(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_img, "People: 0", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_img, "No face detected", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Update auth state - no face means authentication fails
            if self.auth_start_time is not None:
                self.active_auth_user = None
                self.auth_duration = 0
                
            return result_img, face_results
        
        # Create a copy for background blur if needed
        if self.config['blur_background']:
            blurred_img = cv2.GaussianBlur(frame, (25, 25), 0)
            result_img = blurred_img.copy()
        else:
            result_img = frame.copy()
        
        # Process face with MediaPipe for more accurate landmarks if anti-spoofing is enabled
        is_live = True
        if self.config['enable_anti_spoofing'] and self.face_mesh is not None:
            # For performance, mark image as not writeable
            img_rgb.flags.writeable = False
            mp_results = self.face_mesh.process(img_rgb)
            img_rgb.flags.writeable = True
            
            if mp_results and mp_results.multi_face_landmarks:
                face_landmarks = mp_results.multi_face_landmarks[0]  # Use first detected face
                
                # Extract coordinates for easier processing
                landmarks = {}
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks[idx] = (int(landmark.x * frame.shape[1]), 
                                    int(landmark.y * frame.shape[0]))
                
                # Calculate EAR for each eye and use average
                left_ear = self.eye_aspect_ratio(landmarks, self.LEFT_EYE)
                right_ear = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE)
                ear_avg = (left_ear + right_ear) / 2.0
                
                # Store in history
                self.ear_history.append(ear_avg)
                
                # Calibration phase for personalized threshold
                if not self.calibration_complete:
                    self.calibration_frames += 1
                    
                    # After collecting enough samples, calculate baseline
                    if self.calibration_frames >= 30:  # 1 second at 30fps
                        sorted_ears = sorted(self.ear_history)
                        
                        # Use the upper 70% of values for baseline
                        self.baseline_ear = np.mean(sorted_ears[int(len(sorted_ears)*0.3):])
                        self.calibration_complete = True
                        threshold = self.baseline_ear * self.EAR_THRESHOLD_RATIO
                        
                        print(f"Blink calibration complete. Baseline EAR: {self.baseline_ear:.4f}, Threshold: {threshold:.4f}")
                
                # Only proceed with blink detection after calibration
                elif self.calibration_complete:
                    # Calculate dynamic threshold based on baseline
                    threshold = self.baseline_ear * self.EAR_THRESHOLD_RATIO
                    
                    current_time = time.time()
                    
                    if not self.cooldown_active:
                        # Check for blink
                        if ear_avg < threshold and not self.blink_in_progress:
                            self.frames_below_threshold += 1
                            
                            # Confirm blink after minimum frames
                            if self.frames_below_threshold >= self.MIN_BLINK_FRAMES:
                                self.blink_in_progress = True

                        # Check if eyes opened again after blink
                        elif ear_avg > threshold * 1.1 and self.blink_in_progress:
                            self.BLINK_COUNTER += 1
                            self.frames_below_threshold = 0
                            self.blink_in_progress = False
                            self.cooldown_active = True
                            self.last_blink_time = current_time
                            print(f"Blink detected! Count: {self.BLINK_COUNTER}")
                            
                        # Reset counter if not enough consecutive frames
                        elif ear_avg > threshold and self.frames_below_threshold > 0:
                            self.frames_below_threshold = 0
                    else:
                        # Handle cooldown period
                        if current_time - self.last_blink_time > self.BLINK_COOLDOWN:
                            self.cooldown_active = False
                    
                    # If we haven't seen a blink recently, it might be a photo
                    is_live = self.BLINK_COUNTER > 0 and (current_time - self.last_blink_time < 5.0)
                    
                    # Draw eye landmarks for visualization
                    if self.display_ui:
                        if is_live:
                            # Left eye
                            for idx in self.LEFT_EYE:
                                cv2.circle(result_img, landmarks[idx], 2, (0, 255, 0), -1)
                            
                            # Right eye
                            for idx in self.RIGHT_EYE:
                                cv2.circle(result_img, landmarks[idx], 2, (0, 255, 0), -1)
        
        # Prepare batch processing for face encodings
        faces_to_encode = []
        face_locations = []
        
        # Collect all faces for batch processing
        for res in results:
            if res['confidence'] < self.confidence_t:
                continue
                
            face, pt_1, pt_2 = self.get_face(img_rgb, res['box'])
            
            if face.size == 0:
                continue
            
            # Normalize and resize
            face_norm = self.normalize(face)
            face_resized = cv2.resize(face_norm, self.required_size)
            
            # Add to batch
            faces_to_encode.append(face_resized)
            face_locations.append((face, pt_1, pt_2, res))
        
        # Get encodings for all faces at once (if any faces were detected)
        if faces_to_encode:
            face_encodings = self.face_encoder.predict(np.array(faces_to_encode))
            face_encodings = self.l2_normalizer.transform(face_encodings)
        
            # Now process each face with its encoding
            for i, (face, pt_1, pt_2, res) in enumerate(face_locations):
                # Get face encoding and match
                encode = face_encodings[i]
                name = 'unknown'
                distance = float("inf")
                
                for db_name, db_encode in self.encoding_dict.items():
                    dist = cosine(db_encode, encode)
                    if dist < self.recognition_t and dist < distance:
                        name = db_name
                        distance = dist

                # Calculate face region for unblurring
                if self.config['blur_background']:
                    # Copy the unblurred face region back to the result image
                    result_img[pt_1[1]:pt_2[1], pt_1[0]:pt_2[0]] = frame[pt_1[1]:pt_2[1], pt_1[0]:pt_2[0]]
                
                # Update authentication state if we're tracking someone
                recognition_confidence = 1.0 - distance if distance < float("inf") else 0.0
                if name != 'unknown' and is_live:
                    if self.active_auth_user is None:
                        # Start tracking this user
                        self.active_auth_user = name
                        self.auth_start_time = time.time()
                        self.auth_confidence = recognition_confidence
                        self.auth_duration = 0
                        print(f"Starting authentication for {name}")
                    elif self.active_auth_user == name:
                        # Continue tracking this user
                        self.auth_duration = time.time() - self.auth_start_time
                        self.auth_confidence = 0.7 * self.auth_confidence + 0.3 * recognition_confidence
                        
                        # Check if authentication is successful
                        if self.auth_duration >= self.auth_required_time and self.auth_confidence > 0.6:
                            if self.auth_result is None:
                                self.auth_result = True
                                print(f"Authentication successful for {name} (confidence: {self.auth_confidence:.2f})")
                    else:
                        # Different user detected - reset tracking
                        self.active_auth_user = name
                        self.auth_start_time = time.time()
                        self.auth_confidence = recognition_confidence
                        self.auth_duration = 0
                        self.auth_result = None
                        print(f"Switched authentication to {name}")
                elif name == 'unknown' or not is_live:
                    # Reset authentication if tracking but unknown face or not live
                    if self.active_auth_user is not None:
                        print("Face lost or not recognized, resetting authentication")
                        self.active_auth_user = None
                        self.auth_duration = 0
                        self.auth_result = None
                
                # Record detection result
                face_result = {
                    'name': name,
                    'confidence': recognition_confidence,
                    'box': [pt_1[0], pt_1[1], pt_2[0] - pt_1[0], pt_2[1] - pt_1[1]],
                    'is_live': is_live if self.config['enable_anti_spoofing'] else None,
                    'blink_count': self.BLINK_COUNTER if self.config['enable_anti_spoofing'] else None
                }
                face_results.append(face_result)
                
                # Print detection result to terminal
                if name == 'unknown':
                    print(f"Unknown face detected at position {pt_1}")
                else:
                    liveness_text = ""
                    if self.config['enable_anti_spoofing']:
                        liveness_text = ", LIVE" if is_live else ", SPOOF DETECTED!"
                    print(f"Recognized: {name} (confidence: {recognition_confidence:.2f}){liveness_text}")
                
                # Draw results on frame if display is enabled
                if self.display_ui:
                    text_color = (255, 255, 255)  # white text
                    if name == 'unknown':
                        bbox_color = (0, 0, 255)  # red box for unknown
                        cv2.rectangle(result_img, pt_1, pt_2, bbox_color, 2)
                        
                        # Create filled background for text
                        text = name
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(result_img, 
                                    (pt_1[0], pt_1[1] - text_size[1] - 10), 
                                    (pt_1[0] + text_size[0], pt_1[1]), 
                                    bbox_color, -1)
                        cv2.putText(result_img, text, (pt_1[0], pt_1[1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    else:
                        bbox_color = (0, 255, 0) if is_live else (0, 165, 255)  # green if live, orange if potential spoof
                        cv2.rectangle(result_img, pt_1, pt_2, bbox_color, 2)
                        
                        # Create filled background for text
                        conf_text = f"{name} ({recognition_confidence:.2f})"
                        text_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(result_img, 
                                    (pt_1[0], pt_1[1] - text_size[1] - 10), 
                                    (pt_1[0] + text_size[0], pt_1[1]), 
                                    bbox_color, -1)
                        cv2.putText(result_img, conf_text, (pt_1[0], pt_1[1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                        
                        # Add liveness indicator and auth status
                        if self.config['enable_anti_spoofing']:
                            live_text = "LIVE" if is_live else "SPOOF?"
                            live_color = (0, 255, 0) if is_live else (0, 0, 255)
                            
                            # Place liveness indicator below the face
                            cv2.putText(result_img, live_text, 
                                    (pt_1[0], pt_2[1] + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, live_color, 2)
                            
                            # Show blink count
                            cv2.putText(result_img, f"Blinks: {self.BLINK_COUNTER}", 
                                    (pt_1[0], pt_2[1] + 45), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            # Show authentication progress if this is the active user
                            if self.active_auth_user == name:
                                # Calculate progress as percentage of required time
                                auth_progress = min(100, int((self.auth_duration / self.auth_required_time) * 100))
                                
                                # Progress bar position
                                bar_x = pt_1[0]
                                bar_y = pt_2[1] + 70
                                bar_width = pt_2[0] - pt_1[0]
                                bar_height = 10
                                
                                # Draw background
                                cv2.rectangle(result_img, 
                                           (bar_x, bar_y), 
                                           (bar_x + bar_width, bar_y + bar_height), 
                                           (50, 50, 50), -1)
                                
                                # Draw progress
                                progress_width = int(bar_width * auth_progress / 100)
                                cv2.rectangle(result_img, 
                                           (bar_x, bar_y), 
                                           (bar_x + progress_width, bar_y + bar_height), 
                                           (0, 255, 0), -1)
                                
                                # Add percentage text
                                cv2.putText(result_img, f"Auth: {auth_progress}%", 
                                         (bar_x, bar_y - 5), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # Show success message if authenticated
                                if self.auth_result:
                                    cv2.putText(result_img, "AUTHENTICATED", 
                                            (bar_x, bar_y + bar_height + 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Track the recognized face if live
                    if is_live or not self.config['enable_anti_spoofing']:
                        self.recognized_faces.append(name)

        # Calculate and track performance metrics
        process_time = time.time() - process_start_time
        self.process_times.append(process_time)
        avg_process_time = sum(self.process_times) / len(self.process_times)
        
        # Calculate rolling FPS based on moving average of processing times
        self.fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
        
        # Display stats in the UI
        if self.display_ui:
            # Add background for status panel
            cv2.rectangle(result_img, (10, 10), (250, 100), (0, 0, 0, 0.5), -1)
            
            # Display FPS
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(result_img, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show recognized people count
            self.face_count = len(set(self.recognized_faces))
            cv2.putText(result_img, f"People: {self.face_count}", (20, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show auth status if active
            if self.active_auth_user:
                auth_text = f"Authenticating: {self.active_auth_user} ({self.auth_duration:.1f}s)"
                cv2.putText(result_img, auth_text, (20, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result_img, face_results
    
    def authenticate_person(self, person_name=None, max_time=30, live_check=True, display_ui=None):
        """
        Authenticate a specific person or the first recognized person
        
        Args:
            person_name: Name of person to authenticate, or None to authenticate first recognized person
            max_time: Maximum time to wait in seconds
            live_check: Whether to require liveness detection (anti-spoofing)
            display_ui: Override the class's display_ui setting
            
        Returns:
            dict: Authentication result with status information
        """
        if display_ui is not None:
            original_display_ui = self.display_ui
            self.display_ui = display_ui
            
        try:
            # Store the original anti-spoofing setting
            original_anti_spoofing = self.config['enable_anti_spoofing']
            
            # Update configuration
            self.config['enable_anti_spoofing'] = live_check
            
            # Reset the state
            self.reset_state()
            
            # Load models
            if not self.load_models():
                return {"success": False, "authenticated": False, "message": "Failed to load models"}
                
            # Open camera
            if not self.open_video_source():
                return {"success": False, "authenticated": False, "message": "Failed to open camera"}
            
            print(f"Starting authentication{' for ' + person_name if person_name else ''}")
            print("Please look at the camera")
            
            # For authentication timing
            start_time = time.time()
            last_frame_time = start_time
            
            # For UI window
            if self.display_ui:
                cv2.namedWindow("Face Authentication", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Face Authentication", self.width, self.height)
            
            # Authentication loop
            while True:
                # Check for timeout
                current_time = time.time()
                if max_time and current_time - start_time > max_time:
                    print("Authentication timed out")
                    break
                
                # Read frame with retry logic
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame, retrying...")
                    # Try to reconnect to camera
                    self.cap.release()
                    if not self.open_video_source():
                        print("Could not reconnect to camera. Exiting.")
                        break
                    continue
                
                # Process frame for face detection
                result_frame, face_results = self.detect_faces(frame)
                
                # Process authentication results
                if self.auth_result:
                    # Authentication successful
                    authenticated_user = self.active_auth_user
                    
                    # If specific person requested, check if it matches
                    if person_name and authenticated_user != person_name:
                        self.auth_result = False
                        print(f"Wrong person authenticated: {authenticated_user} (expected: {person_name})")
                        continue
                        
                    # Authentication successful
                    print(f"Authentication successful for {authenticated_user}")
                    break
                
                # Display the frame
                if self.display_ui:
                    cv2.imshow("Face Authentication", result_frame)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Authentication cancelled by user")
                        break
                    elif key == ord('r'):
                        print("Resetting authentication")
                        self.reset_state()
                
                # Limit max frame rate to avoid excessive CPU usage
                elapsed = time.time() - last_frame_time
                if elapsed < 0.03:  # No more than ~30 FPS
                    time.sleep(0.03 - elapsed)
                last_frame_time = time.time()
            
            # Clean up
            if self.cap:
                self.cap.release()
            if self.display_ui:
                cv2.destroyAllWindows()
            
            # Prepare result
            authentication_success = self.auth_result == True
            
            # Restore original settings
            self.config['enable_anti_spoofing'] = original_anti_spoofing
            
            # Return authentication result
            return {
                "success": True,
                "authenticated": authentication_success,
                "person": self.active_auth_user if authentication_success else None,
                "confidence": self.auth_confidence if authentication_success else 0,
                "liveness_verified": live_check and self.BLINK_COUNTER > 0,
                "blink_count": self.BLINK_COUNTER,
                "duration": time.time() - start_time,
                "message": "Authentication successful" if authentication_success else "Authentication failed"
            }
            
        except Exception as e:
            print(f"Error during authentication: {e}")
            return {
                "success": False,
                "authenticated": False,
                "message": f"Error during authentication: {str(e)}"
            }
            
        finally:
            # Restore display_ui setting if it was changed
            if display_ui is not None:
                self.display_ui = original_display_ui
    
    def run_recognition(self, max_time=None):
        """
        Run face detection and recognition in a loop
        
        Args:
            max_time: Maximum time to run in seconds (None for indefinite)
            
        Returns:
            dict: Summary of detection session
        """
        if not self.load_models():
            return {"success": False, "message": "Failed to load models"}
            
        if not self.open_video_source():
            return {"success": False, "message": "Failed to open video source"}
            
        # For timing
        start_time = time.time()
        last_frame_time = start_time
        
        # Create window if display is enabled
        if self.display_ui:
            cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Recognition", self.width, self.height)
        
        print("Starting face recognition, press 'q' to quit or 'r' to recalibrate blink detection...")
        
        # For statistics
        total_frames = 0
        face_detected_frames = 0
        detected_names = set()
        
        try:
            while True:
                # Check for timeout
                current_time = time.time()
                if max_time and current_time - start_time > max_time:
                    print(f"Maximum running time of {max_time}s reached, stopping")
                    break
                
                # Read frame with retry logic
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame, retrying...")
                    # Try to reconnect to camera
                    self.cap.release()
                    if not self.open_video_source():
                        print("Could not reconnect to camera. Exiting.")
                        break
                    continue
                    
                # Resize frame if needed
                if frame.shape[1] > self.width or frame.shape[0] > self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                    
                # Process frame
                result_frame, face_results = self.detect_faces(frame)
                
                # Update statistics
                total_frames += 1
                if face_results:
                    face_detected_frames += 1
                    for face in face_results:
                        if face['name'] != 'unknown':
                            detected_names.add(face['name'])
                
                # Display the processed frame
                if self.display_ui:
                    cv2.imshow("Face Recognition", result_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User requested to quit")
                        break
                    elif key == ord('r'):
                        print("Recalibrating blink detection...")
                        # Reset blink detection parameters
                        self.calibration_complete = False
                        self.calibration_frames = 0
                        self.ear_history.clear()
                        self.BLINK_COUNTER = 0
                
                # Limit max frame rate to avoid excessive CPU usage
                elapsed = time.time() - last_frame_time
                if elapsed < 0.03:  # No more than ~30 FPS
                    time.sleep(0.03 - elapsed)
                last_frame_time = time.time()
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error during face recognition: {e}")
        finally:
            # Clean up
            if self.cap:
                self.cap.release()
            if self.display_ui:
                cv2.destroyAllWindows()
        
        # Prepare summary
        total_time = time.time() - start_time
        detection_rate = face_detected_frames / total_frames if total_frames > 0 else 0
        
        summary = {
            "success": True,
            "total_frames": total_frames,
            "frames_with_faces": face_detected_frames,
            "detection_rate": detection_rate,
            "unique_people_detected": list(detected_names),
            "total_people_count": len(detected_names),
            "running_time": total_time,
            "average_fps": total_frames / total_time if total_time > 0 else 0
        }
        
        print("\nDetection Summary:")
        print(f"- Ran for {total_time:.2f} seconds")
        print(f"- Processed {total_frames} frames at {summary['average_fps']:.2f} FPS")
        print(f"- Detected faces in {face_detected_frames} frames ({detection_rate*100:.1f}%)")
        print(f"- Found {len(detected_names)} unique individuals: {', '.join(detected_names) if detected_names else 'None'}")
        
        return summary
    
    def detect_faces_in_image(self, image_path, live_check=False):
        """
        Detect faces in a single image file
        
        Args:
            image_path: Path to image file
            live_check: Whether to check for liveness (always False for still images)
            
        Returns:
            dict: Detection results
        """
        if not self.load_models():
            return {"success": False, "message": "Failed to load models"}
        
        # Load image
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return {"success": False, "message": f"Failed to load image: {image_path}"}
        except Exception as e:
            return {"success": False, "message": f"Error loading image: {e}"}
        
        # Disable anti-spoofing for still images
        original_anti_spoofing = self.config['enable_anti_spoofing']
        self.config['enable_anti_spoofing'] = False
        
        # Process image
        result_frame, face_results = self.detect_faces(frame)
        
        # Restore anti-spoofing setting
        self.config['enable_anti_spoofing'] = original_anti_spoofing
        
        # If display is enabled, show the result
        if self.display_ui:
            cv2.namedWindow("Face Detection Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Face Detection Result", result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Prepare output
        result = {
            "success": True,
            "image_path": image_path,
            "faces_detected": len(face_results),
            "face_details": face_results,
            "unique_people": list(set([face['name'] for face in face_results if face['name'] != 'unknown']))
        }
        
        return result
    
    def __del__(self):
        """Clean up resources when object is deleted"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

# Convenience functions for simpler usage

def recognize_faces(config=None, display_ui=True, max_time=None):
    """
    Convenience function to perform face recognition
    
    Args:
        config: Configuration dictionary or argparse namespace
        display_ui: Whether to display visual interface
        max_time: Maximum time to run in seconds (None for indefinite)
        
    Returns:
        dict: Detection results summary
    """
    recognizer = FaceRecognizer(display_ui=display_ui)
    
    if config:
        # Update with provided config
        if isinstance(config, dict):
            recognizer.config.update(config)
        else:
            # Convert argparse namespace to dict
            for key in vars(config):
                if key in recognizer.config:
                    recognizer.config[key] = getattr(config, key)
    
    return recognizer.run_recognition(max_time=max_time)

def authenticate_person(person_name=None, live_check=True, max_time=30, display_ui=True):
    """
    Authenticate a person using face recognition with optional liveness check
    
    Args:
        person_name: Name of the person to authenticate, or None for any person
        live_check: Whether to require liveness detection (anti-spoofing)
        max_time: Maximum time to wait for authentication in seconds
        display_ui: Whether to show visual feedback during authentication
        
    Returns:
        dict: Authentication results
    """
    recognizer = FaceRecognizer(display_ui=display_ui)
    return recognizer.authenticate_person(
        person_name=person_name,
        live_check=live_check,
        max_time=max_time,
        display_ui=display_ui
    )

def detect_faces_in_image(image_path, config=None, display_ui=True):
    """
    Convenience function to detect faces in a single image
    
    Args:
        image_path: Path to the image file
        config: Configuration dictionary or argparse namespace
        display_ui: Whether to display visual interface
        
    Returns:
        dict: Detection results
    """
    recognizer = FaceRecognizer(display_ui=display_ui)
    
    if config:
        # Update with provided config
        if isinstance(config, dict):
            recognizer.config.update(config)
        else:
            # Convert argparse namespace to dict
            for key in vars(config):
                if key in recognizer.config:
                    recognizer.config[key] = getattr(config, key)
    
    return recognizer.detect_faces_in_image(image_path)

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Face Detection and Recognition')
    parser.add_argument('--encodings', default='encodings/encodings.pkl', help='Path to encodings file')
    parser.add_argument('--model', default='facenet_keras_weights.h5', help='Path to facenet model')
    parser.add_argument('--confidence', type=float, default=0.98, help='Minimum confidence for face detection')
    parser.add_argument('--recognition-threshold', type=float, default=0.4, help='Recognition cosine distance threshold')
    parser.add_argument('--source', default=0, help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--display-size', default='1280x720', help='Display resolution WxH')
    parser.add_argument('--blur-background', action='store_true', help='Blur background around faces')
    parser.add_argument('--enable-anti-spoofing', action='store_true', help='Enable liveness detection')
    parser.add_argument('--headless', action='store_true', help='Run without UI (for API integration)')
    parser.add_argument('--timeout', type=int, help='Maximum running time in seconds')
    parser.add_argument('--image', type=str, help='Process a single image instead of video')
    parser.add_argument('--authenticate', action='store_true', help='Run in authentication mode')
    parser.add_argument('--person', type=str, help='Specific person to authenticate')
    parser.add_argument('--no-liveness', action='store_true', help='Disable liveness check in authentication')
    parser.add_argument('--result-file', type=str, help='Path to write result file')
    
    args = parser.parse_args()
    
    # Create config dictionary from args
    config = {
        'encodings': args.encodings,
        'model': args.model,
        'confidence': args.confidence,
        'recognition_threshold': args.recognition_threshold,
        'source': args.source,
        'display_size': args.display_size,
        'blur_background': args.blur_background,
        'enable_anti_spoofing': args.enable_anti_spoofing
    }
    
    # Run in appropriate mode
    if args.authenticate:
        # Authentication mode
        try:
            # Run authentication with proper error handling
            result = authenticate_person(
                person_name=args.person,
                live_check=not args.no_liveness,
                max_time=args.timeout or 30,
                display_ui=not args.headless
            )
            print(f"Authentication result: {result['message']}")
            
            # Write result to file if specified
            if args.result_file:
                message = f"Authentication {'successful' if result.get('authenticated', False) else 'failed'}\n"
                if result.get('authenticated', False):
                    message += f"Person: {result.get('person', 'unknown')}\n"
                    message += f"Confidence: {result.get('confidence', 0):.2f}\n"
                message += f"Liveness verified: {result.get('liveness_verified', False)}\n"
                message += f"Blink count: {result.get('blink_count', 0)}\n"
                message += f"Message: {result.get('message', 'No message')}"
                write_result_to_file(args.result_file, message)
                
            if result.get('authenticated', False):
                print(f"Authenticated person: {result.get('person')} (confidence: {result.get('confidence', 0):.2f})")
                print(f"Liveness verified: {result.get('liveness_verified', False)} (blinks: {result.get('blink_count', 0)})")
        except Exception as e:
            print(f"Error during authentication: {e}")
            if args.result_file:
                write_result_to_file(args.result_file, f"Authentication failed\nError during authentication: {str(e)}")
    elif args.image:
        # Process a single image
        result = detect_faces_in_image(
            args.image, 
            config=config, 
            display_ui=not args.headless
        )
        print(f"Image processing result: {len(result['face_details'])} faces detected")
        for face in result['face_details']:
            print(f"- {face['name']}: confidence={face['confidence']:.2f}")
    else:
        # Run video processing
        result = recognize_faces(
            config=config, 
            display_ui=not args.headless, 
            max_time=args.timeout
        )
        print(f"Face recognition completed: {result['total_people_count']} people detected")