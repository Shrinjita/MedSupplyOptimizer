import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
from collections import deque
import threading
import base64
from io import BytesIO
from PIL import Image

# Add a global variable to store the current frame
global_frame = None
processing_active = False
last_result = None

# Add a function to get frames for web streaming
def get_jpg_frame():
    global global_frame
    if global_frame is not None:
        # Convert frame to JPEG format for web display
        _, buffer = cv2.imencode('.jpg', global_frame)
        frame_bytes = buffer.tobytes()
        return frame_bytes
    return None

# Add a function to get base64 encoded frame
def get_base64_frame():
    global global_frame
    if global_frame is not None:
        # Convert frame to JPEG format and then to base64
        _, buffer = cv2.imencode('.jpg', global_frame)
        jpg_as_text = base64.b64encode(buffer).decode()
        return jpg_as_text
    return None

class EyeBlinkAuthenticator:
    def __init__(self, display_ui=True):
        # Initialize Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        
        # Eye landmarks for MediaPipe face mesh
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Initialize with optimized parameters for performance
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Blink detection parameters
        self.EAR_THRESHOLD_RATIO = 0.75
        self.MIN_BLINK_FRAMES = 2
        self.MIN_BLINK_TIME = 0.05
        self.BLINK_COOLDOWN = 0.2
        
        # Password system parameters
        self.MAX_SEQUENCE_LENGTH = 8
        self.SEQUENCE_TIMEOUT = 5.0
        
        # Display options
        self.display_ui = display_ui
        
        # Internal state (will be reset at the start of each function)
        self.stored_password = []
        self.reset_state()
    
    def reset_state(self):
        """Reset all internal tracking variables"""
        # State variables - separate for each eye
        self.blink_counter_left = 0
        self.blink_counter_right = 0
        self.blink_counter_both = 0
        self.frames_below_threshold_left = 0
        self.frames_below_threshold_right = 0
        self.last_blink_time_left = time.time()
        self.last_blink_time_right = time.time()
        self.blink_in_progress_left = False
        self.blink_in_progress_right = False
        self.cooldown_active_left = False
        self.cooldown_active_right = False
        
        # Both eyes blink tracking
        self.both_blink_pending = False
        self.both_eyes_closed = False
        self.last_both_blink_time = 0
        self.both_eyes_close_time = 0  # New variable to track when both eyes closed
        self.both_eyes_blink_duration = 0  # Track duration of both-eye blinks
        
        # Password tracking
        self.blink_sequence = []
        self.sequence_start_time = None
        self.password_recording = False
        self.password_verifying = False
        self.password_result = ""
        self.password_entry_cooldown_left = False
        self.password_entry_cooldown_right = False
        
        # History tracking
        self.ear_history_left = deque(maxlen=60)
        self.ear_history_right = deque(maxlen=60)
        self.blink_times_left = deque(maxlen=30)
        self.blink_times_right = deque(maxlen=30)
        self.baseline_ear_left = None
        self.baseline_ear_right = None
        self.calibration_complete = False
        self.calibration_frames = 0
        
        # Eye status for UI
        self.eye_status_left = "Calibrating..."
        self.eye_status_right = "Calibrating..."
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0
    
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Get the eye landmark coordinates
        points = [landmarks[idx] for idx in eye_indices]
        
        # Calculate the vertical distances
        A = dist.euclidean(points[1], points[5])
        B = dist.euclidean(points[2], points[4])
        
        # Calculate the horizontal distance
        C = dist.euclidean(points[0], points[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def calibrate(self, frame, results):
        """Perform eye calibration"""
        if not results.multi_face_landmarks:
            return False
            
        for face_landmarks in results.multi_face_landmarks:
            # Extract coordinates
            landmarks = {}
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks[idx] = (int(landmark.x * frame.shape[1]), 
                                  int(landmark.y * frame.shape[0]))
            
            # Calculate EAR for each eye
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
            
            # Append to history
            self.ear_history_left.append(left_ear)
            self.ear_history_right.append(right_ear)
            
            self.calibration_frames += 1
            
            # After collecting enough samples, calculate baseline
            if self.calibration_frames >= 30:  # 1 second at 30fps
                sorted_ears_left = sorted(self.ear_history_left)
                sorted_ears_right = sorted(self.ear_history_right)
                
                # Use the upper 70% of values for baseline
                self.baseline_ear_left = np.mean(sorted_ears_left[int(len(sorted_ears_left)*0.3):])
                self.baseline_ear_right = np.mean(sorted_ears_right[int(len(sorted_ears_right)*0.3):])
                
                self.calibration_complete = True
                threshold_left = self.baseline_ear_left * self.EAR_THRESHOLD_RATIO
                threshold_right = self.baseline_ear_right * self.EAR_THRESHOLD_RATIO
                
                print(f"Calibration complete:")
                print(f"Left eye - Baseline EAR: {self.baseline_ear_left:.4f}, Threshold: {threshold_left:.4f}")
                print(f"Right eye - Baseline EAR: {self.baseline_ear_right:.4f}, Threshold: {threshold_right:.4f}")
                
                self.eye_status_left = "Open"
                self.eye_status_right = "Open"
                return True
                
        return False
        
    def detect_blinks(self, frame, results, current_time):
        """Core blink detection logic"""
        if not self.calibration_complete or not results.multi_face_landmarks:
            return

        for face_landmarks in results.multi_face_landmarks:
            # Extract coordinates
            landmarks = {}
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks[idx] = (int(landmark.x * frame.shape[1]), 
                                  int(landmark.y * frame.shape[0]))
            
            # Calculate EAR for each eye
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
            
            # Store in history
            self.ear_history_left.append(left_ear)
            self.ear_history_right.append(right_ear)
            
            # Calculate dynamic threshold based on baseline for each eye
            threshold_left = self.baseline_ear_left * self.EAR_THRESHOLD_RATIO
            threshold_right = self.baseline_ear_right * self.EAR_THRESHOLD_RATIO
            
            # Track if both eyes are currently closed (for improved both-eye detection)
            both_eyes_now_closed = (left_ear < threshold_left and right_ear < threshold_right)
            
            # IMPROVED BOTH EYES DETECTION
            # Start tracking a potential both-eye blink when both eyes close simultaneously
            if both_eyes_now_closed and not self.both_eyes_closed:
                self.both_eyes_closed = True
                self.both_blink_pending = True
                self.both_eyes_close_time = current_time
                print("Both eyes closing detected")

            # Complete a both-eye blink when both eyes open after being closed
            if self.both_blink_pending and self.both_eyes_closed:
                if left_ear > threshold_left * 1.1 and right_ear > threshold_right * 1.1:
                    blink_duration = current_time - self.both_eyes_close_time
                    
                    # Only count as a both-eye blink if it's within a reasonable duration range
                    if 0.1 <= blink_duration <= 0.5:
                        self.both_eyes_closed = False
                        self.both_blink_pending = False
                        self.both_eyes_blink_duration = blink_duration
                        
                        # This is a successful both-eyes blink!
                        self.blink_counter_both += 1
                        print(f"Both eyes blinked! Count: {self.blink_counter_both}, Duration: {blink_duration:.3f}s")
                        
                        # Set both eyes on cooldown
                        self.cooldown_active_left = True
                        self.cooldown_active_right = True
                        self.last_blink_time_left = current_time
                        self.last_blink_time_right = current_time
                        self.last_both_blink_time = current_time
                        
                        # Reset individual eye tracking to prevent false counts
                        self.frames_below_threshold_left = 0
                        self.frames_below_threshold_right = 0
                        self.blink_in_progress_left = False
                        self.blink_in_progress_right = False
                        
                        # FIX: Record the blink in sequence if in password mode - ALWAYS ADD TO SEQUENCE
                        if (self.password_recording or self.password_verifying) and len(self.blink_sequence) < self.MAX_SEQUENCE_LENGTH:
                            if self.sequence_start_time is None:
                                self.sequence_start_time = current_time
                            self.blink_sequence.append('B')  # Both eyes
                            # Reset password cooldown flags to prevent phantom blinks
                            self.password_entry_cooldown_left = True
                            self.password_entry_cooldown_right = True
                            print(f"Both eyes blinked - Sequence: {self.blink_sequence}")
                            
                            # Check for password verification
                            if self.password_verifying and len(self.blink_sequence) == len(self.stored_password):
                                if self.blink_sequence == self.stored_password:
                                    self.password_result = "Access Granted!"
                                    print("Password verification successful!")
                                else:
                                    self.password_result = "Access Denied!" 
                                    print("Password verification failed!")
                                self.password_verifying = False
                    else:
                        # Not a valid both-eye blink - either too fast or too slow
                        self.both_eyes_closed = False
                        self.both_blink_pending = False
                        print(f"Invalid both-eye blink duration: {blink_duration:.3f}s")
            
            # If both eyes remained closed for too long, it's not a blink
            elif self.both_blink_pending and self.both_eyes_closed and (current_time - max(self.last_blink_time_left, self.last_blink_time_right) > 1.0):
                self.both_blink_pending = False
                self.both_eyes_closed = False
                print("Both-eyes blink abandoned - eyes held closed too long")
            
            # Update both_eyes_closed tracking if eyes are now open
            if self.both_eyes_closed and (left_ear > threshold_left * 1.1 and right_ear > threshold_right * 1.1):
                self.both_eyes_closed = False
            
            # Process left eye
            if not self.cooldown_active_left and not self.both_blink_pending:
                # Check for blink
                if left_ear < threshold_left and not self.blink_in_progress_left:
                    self.frames_below_threshold_left += 1
                    self.eye_status_left = "Closing"
                    
                    # Confirm blink after minimum frames
                    if self.frames_below_threshold_left >= self.MIN_BLINK_FRAMES:
                        if current_time - self.last_blink_time_left > self.MIN_BLINK_TIME:
                            self.blink_in_progress_left = True
                            self.eye_status_left = "Blink!"

                # Check if eye opened again after potential blink
                elif left_ear > threshold_left * 1.1 and self.blink_in_progress_left:
                    # Only count individual blinks if not part of a both-eyes blink
                    if not self.both_blink_pending:
                        self.blink_counter_left += 1
                        self.blink_times_left.append(current_time)
                        self.last_blink_time_left = current_time
                        
                        # FIX: Make sure single eye blinks are always recorded in the sequence
                        if (self.password_recording or self.password_verifying) and len(self.blink_sequence) < self.MAX_SEQUENCE_LENGTH:
                            # Make sure enough time passed since last blink of any kind
                            if current_time - max(self.last_blink_time_right, self.last_both_blink_time) > 0.5:
                                if self.sequence_start_time is None:
                                    self.sequence_start_time = current_time
                                self.blink_sequence.append('L')  # Left eye
                                print(f"Left eye blinked - Sequence: {self.blink_sequence}")
                        
                    self.frames_below_threshold_left = 0
                    self.blink_in_progress_left = False
                    self.cooldown_active_left = True
                    
                # Reset counter if not enough consecutive frames
                elif left_ear > threshold_left and self.frames_below_threshold_left > 0:
                    self.frames_below_threshold_left = 0
                    self.eye_status_left = "Open"
            else:
                # Handle cooldown period
                if current_time - self.last_blink_time_left > self.BLINK_COOLDOWN:
                    self.cooldown_active_left = False
                    self.eye_status_left = "Open"
                    
                # Reset password cooldown if enough time has passed
                if self.password_entry_cooldown_left and current_time - self.last_blink_time_left > 1.0:
                    self.password_entry_cooldown_left = False

            # Process right eye (same structure but with fixes)
            if not self.cooldown_active_right and not self.both_blink_pending:
                # Check for blink
                if right_ear < threshold_right and not self.blink_in_progress_right:
                    self.frames_below_threshold_right += 1
                    self.eye_status_right = "Closing"
                    
                    # Confirm blink after minimum frames
                    if self.frames_below_threshold_right >= self.MIN_BLINK_FRAMES:
                        if current_time - self.last_blink_time_right > self.MIN_BLINK_TIME:
                            self.blink_in_progress_right = True
                            self.eye_status_right = "Blink!"

                # Check if eye opened again after potential blink
                elif right_ear > threshold_right * 1.1 and self.blink_in_progress_right:
                    # Only count individual blinks if not part of a both-eyes blink
                    if not self.both_blink_pending:
                        self.blink_counter_right += 1
                        self.blink_times_right.append(current_time)
                        self.last_blink_time_right = current_time
                        
                        # FIX: Make sure single eye blinks are always recorded in the sequence
                        if (self.password_recording or self.password_verifying) and len(self.blink_sequence) < self.MAX_SEQUENCE_LENGTH:
                            # Make sure enough time passed since last blink of any kind
                            if current_time - max(self.last_blink_time_left, self.last_both_blink_time) > 0.5:
                                if self.sequence_start_time is None:
                                    self.sequence_start_time = current_time
                                self.blink_sequence.append('R')  # Right eye
                                print(f"Right eye blinked - Sequence: {self.blink_sequence}")
                        
                    self.frames_below_threshold_right = 0
                    self.blink_in_progress_right = False
                    self.cooldown_active_right = True
                    
                # Reset counter if not enough consecutive frames
                elif right_ear > threshold_right and self.frames_below_threshold_right > 0:
                    self.frames_below_threshold_right = 0
                    self.eye_status_right = "Open"
            else:
                # Handle cooldown period
                if current_time - self.last_blink_time_right > self.BLINK_COOLDOWN:
                    self.cooldown_active_right = False
                    self.eye_status_right = "Open"
                    
                # Reset password cooldown if enough time has passed
                if self.password_entry_cooldown_right and current_time - self.last_blink_time_right > 1.0:
                    self.password_entry_cooldown_right = False

            # Check for password verification completion
            if self.password_verifying and len(self.blink_sequence) == len(self.stored_password):
                if self.blink_sequence == self.stored_password:
                    self.password_result = "Access Granted!"
                    print("Password verification successful!")
                else:
                    self.password_result = "Access Denied!"
                    print("Password verification failed!")
                self.password_verifying = False
                
            return landmarks  # Return landmarks for UI display
        
        return None
        
    def draw_ui(self, frame, landmarks):
        """Draw UI elements for visualization"""
        if not self.display_ui or not landmarks:
            return frame
            
        # Draw eye landmarks for visualization
        # Left eye (green)
        for idx in self.LEFT_EYE:
            cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
        left_eye_points = [landmarks[idx] for idx in self.LEFT_EYE]
        left_eye_contour = np.array(left_eye_points, dtype=np.int32)
        cv2.polylines(frame, [left_eye_contour], True, (0, 255, 0), 1)
        
        # Right eye (blue)
        for idx in self.RIGHT_EYE:
            cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)
        right_eye_points = [landmarks[idx] for idx in self.RIGHT_EYE]
        right_eye_contour = np.array(right_eye_points, dtype=np.int32)
        cv2.polylines(frame, [right_eye_contour], True, (255, 0, 0), 1)
        
        # Draw status and metrics
        cv2.rectangle(frame, (10, 10), (320, 170), (0, 0, 0), -1)
        
        # Status colors
        left_status_color = (0, 255, 0) if self.eye_status_left == "Open" else (0, 0, 255)
        right_status_color = (255, 0, 0) if self.eye_status_right == "Open" else (0, 0, 255)
        
        # Eye status text
        cv2.putText(frame, f"Left Eye: {self.eye_status_left}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_status_color, 2)
        cv2.putText(frame, f"Right Eye: {self.eye_status_right}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_status_color, 2)
        
        # Display blink counts
        cv2.putText(frame, f"Left Blinks: {self.blink_counter_left}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Right Blinks: {self.blink_counter_right}", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Both Eyes: {self.blink_counter_both}", (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display password mode info if active
        if self.password_recording or self.password_verifying or self.stored_password:
            password_panel_y = 180
            cv2.rectangle(frame, (10, password_panel_y), (320, password_panel_y + 120), (20, 20, 20), -1)
            
            # Mode display
            mode_text = "Recording Password" if self.password_recording else "Verifying Password" if self.password_verifying else "Password System"
            cv2.putText(frame, mode_text, (20, password_panel_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Current sequence
            if self.password_recording or self.password_verifying:
                sequence_str = ''.join(self.blink_sequence)
                remaining_time = max(0, self.SEQUENCE_TIMEOUT - (time.time() - self.sequence_start_time)) if self.sequence_start_time else 0
                cv2.putText(frame, f"Sequence: {sequence_str}", (20, password_panel_y + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Time left: {remaining_time:.1f}s", (20, password_panel_y + 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Stored password
            if self.stored_password:
                stored_str = ''.join(self.stored_password)
                cv2.putText(frame, f"Stored pass: {stored_str}", (20, password_panel_y + 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 255), 2)
            
            # Verification result
            if self.password_result:
                result_color = (0, 255, 0) if "Granted" in self.password_result else (0, 0, 255)
                cv2.putText(frame, self.password_result, (frame.shape[1] - 300, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                            
        return frame

    def check_timeout(self, current_time):
        """Check for sequence timeout"""
        if (self.password_recording or self.password_verifying) and self.sequence_start_time and \
            current_time - self.sequence_start_time > self.SEQUENCE_TIMEOUT:
            if self.password_recording:
                self.stored_password = self.blink_sequence.copy()
                print(f"Password recorded: {self.stored_password}")
                self.password_recording = False
            elif self.password_verifying:
                self.password_result = "Timeout - Verification Failed"
                self.password_verifying = False
            self.blink_sequence = []
            self.sequence_start_time = None
            return True
        return False
        
    def set_password(self, max_time=None):
        """Record a password sequence and return it"""
        global global_frame, processing_active
        
        print("Starting password recording...")
        self.reset_state()
        processing_active = True
        
        # Set up camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set password recording mode
        self.password_recording = True
        
        start_time = time.time()
        
        # Create only one window that will be used for the entire process
        if self.display_ui:
            cv2.namedWindow("Eye Blink Password System")
        
        # First do calibration in the background (no separate window)
        print("Calibrating... Keep eyes open and look at the camera")
        while not self.calibration_complete:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                processing_active = False
                return None
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Try to calibrate
            calibration_completed = self.calibrate(frame, results)
            
            # Draw UI elements on frame
            if results.multi_face_landmarks:
                # Draw face mesh with MediaPipe
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
                    )
                    
                    # Highlight eye landmarks
                    landmarks = {}
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        landmarks[idx] = (int(landmark.x * frame.shape[1]), 
                                        int(landmark.y * frame.shape[0]))
                                        
                    # Draw eye landmarks for visualization
                    # Left eye (green)
                    for idx in self.LEFT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
                    
                    # Right eye (blue)  
                    for idx in self.RIGHT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)
                    
            # Add calibration text
            cv2.putText(frame, "Calibrating... Keep eyes open", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update global_frame regardless of display_ui setting
            global_frame = frame.copy()
            
            # Show the single window
            if self.display_ui:
                cv2.imshow("Eye Blink Password System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    processing_active = False
                    return None
            
            # Check for timeout
            if max_time and time.time() - start_time > max_time:
                print("Calibration timeout")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                processing_active = False
                return None
        
        print("Calibration complete! Now countdown before recording...")
        
        # Add countdown from 5 to 0
        countdown_start = time.time()
        while time.time() - countdown_start < 5:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Calculate remaining seconds
            remaining = 5 - int(time.time() - countdown_start)
            
            # Process frame with face mesh to show eye tracking
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw face landmarks
            if results.multi_face_landmarks:
                landmarks = {}
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
                    )
                    
                    # Extract landmarks for eye visualization
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        landmarks[idx] = (int(landmark.x * frame.shape[1]), 
                                        int(landmark.y * frame.shape[0]))
                    
                    # Draw eye landmarks
                    for idx in self.LEFT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
                    for idx in self.RIGHT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)
            
            # Add countdown text
            cv2.putText(frame, f"Get Ready! Recording starts in {remaining}...", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Prepare to record your eye blink pattern", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update global frame
            global_frame = frame.copy()
            
            # Show the same window
            if self.display_ui:
                cv2.imshow("Eye Blink Password System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    processing_active = False
                    return None
        
        print("Countdown complete! Now record your password by blinking...")
        print("Use left eye (L), right eye (R), or both eyes (B) blinks")
        
        # Now start password recording using the same window
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            current_time = time.time()
            
            # Check for timeout
            if max_time and current_time - start_time > max_time:
                print("Password recording timeout")
                if self.stored_password:
                    cap.release()
                    if self.display_ui:
                        cv2.destroyAllWindows()
                    processing_active = False
                    return self.stored_password
                else:
                    cap.release()
                    if self.display_ui:
                        cv2.destroyAllWindows()
                    processing_active = False
                    return None
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Check for sequence timeout
            timeout_occurred = self.check_timeout(current_time)
            
            # Process blinks
            landmarks = self.detect_blinks(frame, results, current_time)
            
            # Draw UI elements
            frame_with_ui = self.draw_ui(frame, landmarks)
            
            # Update global_frame
            global_frame = frame_with_ui.copy()
            
            # Display UI in the same window
            if self.display_ui:
                cv2.imshow("Eye Blink Password System", frame_with_ui)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    processing_active = False
                    return None
            
            # If password recording is complete, return the password
            if not self.password_recording and self.stored_password:
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                print(f"Password recorded: {''.join(self.stored_password)}")
                processing_active = False
                return self.stored_password
        
        # Cleanup
        cap.release()
        if self.display_ui:
            cv2.destroyAllWindows()
        
        processing_active = False
        
        # Return the password if it was recorded
        if self.stored_password:
            return self.stored_password
        return None
            
    def verify_password(self, stored_password, max_time=None):
        """Verify a password sequence"""
        global global_frame, processing_active
        
        print("Starting password verification...")
        self.reset_state()
        self.stored_password = stored_password
        processing_active = True
        
        # Set up camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        
        # Create only one window for the entire verification process
        if self.display_ui:
            cv2.namedWindow("Eye Blink Password System")
        
        # First do calibration
        print("Calibrating... Keep eyes open and look at the camera")
        while not self.calibration_complete:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                processing_active = False
                return False
                
            global_frame = frame.copy()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Try to calibrate
            calibration_completed = self.calibrate(frame, results)
            
            # Draw UI elements
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
                    )
                    
                    # Extract landmarks
                    landmarks = {}
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        landmarks[idx] = (int(landmark.x * frame.shape[1]), 
                                        int(landmark.y * frame.shape[0]))
                    
                    # Draw eye highlights
                    for idx in self.LEFT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
                    for idx in self.RIGHT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)
            
            # Add calibration text
            cv2.putText(frame, "Calibrating... Keep eyes open", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update global frame
            global_frame = frame.copy()
            
            if self.display_ui:
                cv2.imshow("Eye Blink Password System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    processing_active = False
                    return False
            
            # Check for timeout
            if max_time and time.time() - start_time > max_time:
                print("Calibration timeout")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                processing_active = False
                return False
        
        # Add countdown from 5 to 0
        countdown_start = time.time()
        while time.time() - countdown_start < 5:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Calculate remaining seconds
            remaining = 5 - int(time.time() - countdown_start)
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw face landmarks
            if results.multi_face_landmarks:
                landmarks = {}
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
                    )
                    
                    # Extract landmarks
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        landmarks[idx] = (int(landmark.x * frame.shape[1]), 
                                        int(landmark.y * frame.shape[0]))
                    
                    # Draw eye landmarks
                    for idx in self.LEFT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
                    for idx in self.RIGHT_EYE:
                        cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)
            
            # Add countdown text
            cv2.putText(frame, f"Get Ready! Verification starts in {remaining}...", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Enter your {len(stored_password)}-blink pattern", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update global frame
            global_frame = frame.copy()
            
            # Show the same window
            if self.display_ui:
                cv2.imshow("Eye Blink Password System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    processing_active = False
                    return False
        
        print("Calibration complete! Now enter your password by blinking...")
        print(f"Enter a {len(stored_password)}-blink sequence")
        
        # Start password verification
        self.password_verifying = True
        
        # Now start verification using the same window
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            global_frame = frame.copy()
            
            current_time = time.time()
            
            # Check for timeout
            if max_time and current_time - start_time > max_time:
                print("Password verification timeout")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                processing_active = False
                return False
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Check for sequence timeout
            timeout_occurred = self.check_timeout(current_time)
            
            # Process blinks
            landmarks = self.detect_blinks(frame, results, current_time)
            
            # Draw UI
            frame_with_ui = self.draw_ui(frame, landmarks)
            global_frame = frame_with_ui.copy()
            
            # Display in single window
            if self.display_ui:
                cv2.imshow("Eye Blink Password System", frame_with_ui)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    processing_active = False
                    return False
            
            # If password verification is complete, return the result
            if not self.password_verifying:
                verification_successful = "Granted" in self.password_result
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                processing_active = False
                return verification_successful

        # Cleanup
        cap.release()
        if self.display_ui:
            cv2.destroyAllWindows()

        processing_active = False

        # Verification failed if we get here
        return False

    def record_password_headless(self, max_time=30, display_ui=None):
        """
        Record a blink password without requiring UI interaction, suitable for API/frontend integration.
        
        Args:
            max_time: Maximum time for the whole process in seconds
            display_ui: Override the class's display_ui setting
            
        Returns:
            dict: A result dictionary with status information and the password
        """
        if display_ui is not None:
            original_display_ui = self.display_ui
            self.display_ui = display_ui
        
        try:
            password = self.set_password(max_time=max_time)
            
            if password:
                return {
                    "success": True,
                    "message": "Password successfully recorded",
                    "password": password,
                    "password_string": ''.join(password),
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to record password",
                    "password": None
                }
        finally:
            # Restore original display_ui setting if it was changed
            if display_ui is not None:
                self.display_ui = original_display_ui

    def verify_password_headless(self, stored_password, max_time=30, display_ui=None):
        """
        Verify a blink password without requiring UI interaction, suitable for API/frontend integration.
        
        Args:
            stored_password: The password sequence to verify against (list or string)
            max_time: Maximum time for the whole process in seconds
            display_ui: Override the class's display_ui setting
            
        Returns:
            dict: A result dictionary with verification status information
        """
        if display_ui is not None:
            original_display_ui = self.display_ui
            self.display_ui = display_ui
        
        # Convert string password to list if needed
        if isinstance(stored_password, str):
            password_list = list(stored_password)
        else:
            password_list = stored_password
            
        try:
            verification_result = self.verify_password(password_list, max_time=max_time)
            
            if verification_result:
                return {
                    "success": True,
                    "verified": True,
                    "message": "Password verification successful"
                }
            else:
                return {
                    "success": True,
                    "verified": False,
                    "message": "Password verification failed"
                }
        except Exception as e:
            return {
                "success": False,
                "verified": False,
                "message": f"Error during password verification: {str(e)}"
            }
        finally:
            # Restore original display_ui setting if it was changed
            if display_ui is not None:
                self.display_ui = original_display_ui


def authenticate_user(stored_password=None, max_time=30, display_ui=True):
    global global_frame, processing_active, last_result
    processing_active = True
    
    authenticator = EyeBlinkAuthenticator(display_ui=display_ui)
    
    if stored_password is None:
        result = authenticator.record_password_headless(max_time=max_time)
    else:
        result = authenticator.verify_password_headless(stored_password, max_time=max_time)
    
    processing_active = False
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Eye Blink Authentication System")
    parser.add_argument('--headless', action='store_true', help='Run without UI (for API integration)')
    parser.add_argument('--record', action='store_true', help='Record a new password')
    parser.add_argument('--verify', type=str, help='Verify against the provided password string')
    parser.add_argument('--timeout', type=int, default=30, help='Maximum time to wait in seconds')
    
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.record:
        result = authenticate_user(stored_password=None, max_time=args.timeout, display_ui=not args.headless)
        print(result)
    elif args.verify:
        result = authenticate_user(stored_password=args.verify, max_time=args.timeout, display_ui=not args.headless)
        print(result)
    else:
        # Interactive mode
        authenticator = EyeBlinkAuthenticator(display_ui=True)
        
        print("Eye Blink Authentication System")
        print("1. Record a new password")
        print("2. Verify a password")
        print("q. Quit")
        
        choice = input("Choose an option: ")
        
        if choice == '1':
            password = authenticator.set_password()
            if password:
                print(f"Password recorded: {''.join(password)}")
        elif choice == '2':
            password_str = input("Enter the password to verify against: ")
            result = authenticator.verify_password(list(password_str))
            print(f"Verification result: {'Success' if result else 'Failed'}")
        else:
            print("Exiting...")
