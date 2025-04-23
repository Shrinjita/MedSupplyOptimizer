"""
neck_gesture_detection.py - Advanced neck gesture recognition system with sequence authentication
Supports up, down, left, right movements with improved sensitivity and angle-invariant detection
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import argparse

class NeckGestureDetector:
    def __init__(self, display_ui=True):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Important landmark indices
        self.NOSE = 0
        self.LEFT_EYE = 2
        self.RIGHT_EYE = 5
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        
        # Key points for tracking
        self.KEY_POINTS = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, 
                         self.LEFT_EAR, self.RIGHT_EAR, 
                         self.LEFT_SHOULDER, self.RIGHT_SHOULDER]
        
        # Gesture thresholds - made more sensitive
        self.THRESHOLDS = {
            'vertical': 0.08,     # Up/Down threshold (was 0.15)
            'horizontal': 0.06,   # Left/Right threshold (was 0.10)
            'nod_speed': 0.5      # Speed threshold for quick nods/shakes
        }
        
        # Detection parameters
        self.DEAD_ZONE = 0.4            # % of threshold to use as dead zone (smaller for more sensitivity)
        self.MIN_GESTURE_DURATION = 0.2  # Seconds
        self.COOLDOWN_TIME = 0.5        # Seconds between gestures
        self.SEQUENCE_TIMEOUT = 7.0     # Seconds for sequence input
        self.MAX_SEQUENCE_LENGTH = 8    # Maximum gesture sequence length
        
        # Display options
        self.display_ui = display_ui
        
        # Internal state
        self.reset_state()
    
    def reset_state(self):
        """Reset all tracking variables"""
        # Calibration data
        self.calibration = {
            'frames': 0,
            'is_complete': False,
            'neck_positions': [],
            'head_angles': []
        }
        
        # Baseline values
        self.baseline = {
            'neck_position': None,     # Normalized neck position
            'head_angle': None,        # Head angle
            'shoulder_width': None,    # Distance between shoulders
            'eye_distance': None       # Distance between eyes
        }
        
        # Gesture state
        self.current_gesture = None
        self.last_gesture = None
        self.gesture_start_time = 0
        self.last_gesture_time = 0
        self.gesture_in_progress = False
        self.cooldown_active = False
        
        # Gesture tracking for sequences
        self.gesture_count = {
            'Up': 0,
            'Down': 0,
            'Left': 0,
            'Right': 0
        }
        
        # Sequence tracking
        self.blink_sequence = []
        self.sequence_start_time = None
        self.password_recording = False
        self.password_verifying = False
        self.stored_password = []
        self.password_result = ""
        
        # Position history for movement detection
        self.position_history = {
            'vertical': deque(maxlen=15),
            'horizontal': deque(maxlen=15),
            'timestamps': deque(maxlen=15)
        }
        
        # FPS tracking
        self.prev_frame_time = 0
        self.fps = 0
        
        # Debug metrics
        self.metrics = {
            'vertical_displacement': 0,
            'horizontal_displacement': 0,
            'in_dead_zone': True
        }
    
    def calibrate(self, landmarks):
        """Establish baseline neutral position that works with any camera angle"""
        if self.calibration['is_complete']:
            return True
        
        # Get key points
        nose = np.array([landmarks[self.NOSE].x, landmarks[self.NOSE].y])
        left_eye = np.array([landmarks[self.LEFT_EYE].x, landmarks[self.LEFT_EYE].y])
        right_eye = np.array([landmarks[self.RIGHT_EYE].x, landmarks[self.RIGHT_EYE].y])
        left_ear = np.array([landmarks[self.LEFT_EAR].x, landmarks[self.LEFT_EAR].y])
        right_ear = np.array([landmarks[self.RIGHT_EAR].x, landmarks[self.RIGHT_EAR].y])
        left_shoulder = np.array([landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y])
        
        # Calculate midpoints and reference measurements
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        eye_mid = (left_eye + right_eye) / 2
        ear_mid = (left_ear + right_ear) / 2
        
        # Normalize neck position by shoulder width (angle-invariant)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Calculate head-to-shoulder vector (angle-invariant relative position)
        neck_vector = nose - shoulder_mid
        normalized_neck_pos = neck_vector / max(shoulder_width, 0.001)  # Prevent division by zero
        
        # Calculate head angle using ear vector (works at any camera angle)
        ear_vector = right_ear - left_ear
        head_angle = np.degrees(np.arctan2(ear_vector[1], ear_vector[0]))
        
        # Store calibration data
        self.calibration['neck_positions'].append(normalized_neck_pos)
        self.calibration['head_angles'].append(head_angle)
        self.calibration['frames'] += 1
        
        # After collecting enough samples, establish baseline
        if self.calibration['frames'] >= 30:  # 1 second at 30fps
            # Use median values for robustness
            neck_positions = np.array(self.calibration['neck_positions'])
            head_angles = np.array(self.calibration['head_angles'])
            
            # Set baseline using median values
            self.baseline['neck_position'] = np.median(neck_positions, axis=0)
            self.baseline['head_angle'] = np.median(head_angles)
            self.baseline['shoulder_width'] = shoulder_width
            self.baseline['eye_distance'] = eye_distance
            
            # Mark calibration as complete
            self.calibration['is_complete'] = True
            print("Calibration complete:")
            print(f"Baseline head position: {self.baseline['neck_position']}")
            print(f"Baseline head angle: {self.baseline['head_angle']:.2f}°")
            return True
        
        return False
    
    def detect_gesture(self, landmarks, current_time):
        """Detect neck gestures from landmarks with improved sensitivity"""
        if not self.calibration['is_complete']:
            return None
        
        # Get key points
        nose = np.array([landmarks[self.NOSE].x, landmarks[self.NOSE].y])
        left_eye = np.array([landmarks[self.LEFT_EYE].x, landmarks[self.LEFT_EYE].y])
        right_eye = np.array([landmarks[self.RIGHT_EYE].x, landmarks[self.RIGHT_EYE].y])
        left_ear = np.array([landmarks[self.LEFT_EAR].x, landmarks[self.LEFT_EAR].y])
        right_ear = np.array([landmarks[self.RIGHT_EAR].x, landmarks[self.RIGHT_EAR].y])
        left_shoulder = np.array([landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y])
        
        # Calculate midpoints and measurements
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        eye_mid = (left_eye + right_eye) / 2
        
        # Normalize by current shoulder width for distance invariance
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        scale_factor = self.baseline['shoulder_width'] / max(shoulder_width, 0.001)
        
        # Calculate normalized neck position
        neck_vector = nose - shoulder_mid
        normalized_neck_pos = neck_vector / max(shoulder_width, 0.001)
        
        # Calculate displacements (angle-invariant and distance-invariant)
        vertical_displacement = (normalized_neck_pos[1] - self.baseline['neck_position'][1]) * scale_factor
        horizontal_displacement = (normalized_neck_pos[0] - self.baseline['neck_position'][0]) * scale_factor
        
        # Store history for movement detection
        self.position_history['vertical'].append(vertical_displacement)
        self.position_history['horizontal'].append(horizontal_displacement)
        self.position_history['timestamps'].append(current_time)
        
        # Apply exponential smoothing for stability while maintaining sensitivity
        if len(self.position_history['vertical']) >= 3:
            # More weight to recent values for responsiveness
            weights = np.linspace(0.5, 1.0, len(self.position_history['vertical']))
            weights = weights / np.sum(weights)
            
            # Weighted averages
            smooth_vertical = np.average(self.position_history['vertical'], weights=weights)
            smooth_horizontal = np.average(self.position_history['horizontal'], weights=weights)
        else:
            smooth_vertical = vertical_displacement
            smooth_horizontal = horizontal_displacement
        
        # Update metrics for UI display
        self.metrics['vertical_displacement'] = smooth_vertical
        self.metrics['horizontal_displacement'] = smooth_horizontal
        
        # Calculate dead zones (smaller for increased sensitivity)
        vertical_dead_zone = self.THRESHOLDS['vertical'] * self.DEAD_ZONE
        horizontal_dead_zone = self.THRESHOLDS['horizontal'] * self.DEAD_ZONE
        
        # Check if in dead zone
        in_dead_zone = (abs(smooth_vertical) < vertical_dead_zone and 
                       abs(smooth_horizontal) < horizontal_dead_zone)
        
        self.metrics['in_dead_zone'] = in_dead_zone
        
        # Reset gesture tracking when returning to neutral position
        if in_dead_zone:
            if self.gesture_in_progress:
                self.gesture_in_progress = False
                self.current_gesture = None  # Reset current gesture when returning to neutral
                self.cooldown_active = False  # Reset cooldown when returning to neutral
            return None
        
        # Determine gesture based on displacement
        gesture = None
        
        # Prioritize larger movement axis for more intuitive detection
        if abs(smooth_vertical) > abs(smooth_horizontal):
            # Vertical movement is dominant
            if smooth_vertical < -self.THRESHOLDS['vertical']:
                gesture = "Up"
            elif smooth_vertical > self.THRESHOLDS['vertical']:
                gesture = "Down"
        else:
            # Horizontal movement is dominant
            if smooth_horizontal < -self.THRESHOLDS['horizontal']:
                gesture = "Left"
            elif smooth_horizontal > self.THRESHOLDS['horizontal']:
                gesture = "Right"
        
        # Check if this is a new gesture movement
        if gesture != self.current_gesture:
            # Reset cooldown for a different gesture - THIS IS THE KEY CHANGE
            if gesture is not None:
                self.cooldown_active = False
                self.gesture_in_progress = False
                
            self.current_gesture = gesture
            self.gesture_start_time = current_time
            self.gesture_in_progress = True
            return None
        
        # Only register a gesture once it has been held for MIN_GESTURE_DURATION
        # and only if we haven't already recorded this specific gesture movement
        if (self.gesture_in_progress and gesture and 
            current_time - self.gesture_start_time >= self.MIN_GESTURE_DURATION and
            not self.cooldown_active):
            
            # Record gesture
            self.gesture_count[gesture] += 1
            self.last_gesture = gesture
            self.last_gesture_time = current_time
            self.cooldown_active = True  # Prevent repeated detections until returning to neutral or changing gesture
            
            # Handle sequence recording/verification
            if (self.password_recording or self.password_verifying) and len(self.blink_sequence) < self.MAX_SEQUENCE_LENGTH:
                if self.sequence_start_time is None:
                    self.sequence_start_time = current_time
                
                # Map gestures to codes like the eye blink system
                gesture_code = self._gesture_to_code(gesture)
                self.blink_sequence.append(gesture_code)
                print(f"Gesture detected: {gesture} ({gesture_code}) - Sequence: {self.blink_sequence}")
                
                # Check verification completion
                if self.password_verifying and len(self.blink_sequence) == len(self.stored_password):
                    if self.blink_sequence == self.stored_password:
                        self.password_result = "Access Granted!"
                        print("Password verification successful!")
                    else:
                        self.password_result = "Access Denied!"
                        print("Password verification failed!")
                    self.password_verifying = False
            
            return gesture
                
        return None
    
    def _gesture_to_code(self, gesture):
        """Convert gesture name to a single character code for sequences"""
        mapping = {
            'Up': 'U',
            'Down': 'D',
            'Left': 'L',
            'Right': 'R'
        }
        return mapping.get(gesture, 'X')  # X for unknown
    
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
    
    def process_frame(self, frame):
        """Process a video frame to detect neck gestures"""
        # Calculate FPS
        current_time = time.time()
        if self.prev_frame_time > 0:
            self.fps = 1 / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # Check if pose was detected
        if not results.pose_landmarks:
            if self.display_ui:
                self._draw_ui(frame, None, None, "No pose detected")
            return frame, None
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Run calibration if needed
        if not self.calibration['is_complete']:
            self.calibrate(landmarks)
            if self.display_ui:
                self._draw_ui(frame, landmarks, None, "Calibrating... Please look straight ahead")
            return frame, None
        
        # Check for sequence timeout
        self.check_timeout(current_time)
        
        # Detect gesture
        gesture = self.detect_gesture(landmarks, current_time)
        
        # Draw UI
        if self.display_ui:
            self._draw_ui(frame, landmarks, gesture)
        
        return frame, gesture
    
    def _draw_ui(self, frame, landmarks, gesture, status_text=None):
        """Draw visualization UI with detailed feedback"""
        if not self.display_ui:
            return frame
        
        height, width = frame.shape[:2]
        
        # Draw landmarks if available
        if landmarks:
            # Draw skeleton for key points
            points = {}
            for idx in self.KEY_POINTS:
                x, y = int(landmarks[idx].x * width), int(landmarks[idx].y * height)
                points[idx] = (x, y)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections
            cv2.line(frame, points[self.NOSE], 
                    ((points[self.LEFT_SHOULDER][0] + points[self.RIGHT_SHOULDER][0]) // 2,
                     (points[self.LEFT_SHOULDER][1] + points[self.RIGHT_SHOULDER][1]) // 2), 
                    (0, 255, 0), 2)
            cv2.line(frame, points[self.LEFT_SHOULDER], points[self.RIGHT_SHOULDER], (0, 255, 0), 2)
            cv2.line(frame, points[self.LEFT_EYE], points[self.RIGHT_EYE], (0, 255, 0), 2)
            cv2.line(frame, points[self.LEFT_EAR], points[self.LEFT_EYE], (0, 255, 0), 2)
            cv2.line(frame, points[self.RIGHT_EAR], points[self.RIGHT_EYE], (0, 255, 0), 2)
            cv2.line(frame, points[self.LEFT_EYE], points[self.NOSE], (0, 255, 0), 2)
            cv2.line(frame, points[self.RIGHT_EYE], points[self.NOSE], (0, 255, 0), 2)
        
        # Status panel
        cv2.rectangle(frame, (10, 10), (320, 170), (0, 0, 0), -1)
        
        # Display status text or detected gesture
        if status_text:
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if self.current_gesture and self.gesture_in_progress:
                cv2.putText(frame, f"Detecting: {self.current_gesture}", (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No gesture detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display gesture counts
        cv2.putText(frame, f"Up: {self.gesture_count['Up']}", (20, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Down: {self.gesture_count['Down']}", (20, 100),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Left: {self.gesture_count['Left']}", (150, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Right: {self.gesture_count['Right']}", (150, 100),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show displacement values
        v_color = (0, 255, 0) if abs(self.metrics['vertical_displacement']) > self.THRESHOLDS['vertical'] else (255, 255, 255)
        h_color = (0, 255, 0) if abs(self.metrics['horizontal_displacement']) > self.THRESHOLDS['horizontal'] else (255, 255, 255)
        
        cv2.putText(frame, f"V: {self.metrics['vertical_displacement']:.3f}", (20, 130), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, v_color, 2)
        cv2.putText(frame, f"H: {self.metrics['horizontal_displacement']:.3f}", (150, 130), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 2)
        
        # Thresholds info
        cv2.putText(frame, f"Thresholds: V={self.THRESHOLDS['vertical']:.2f}, H={self.THRESHOLDS['horizontal']:.2f}", 
                  (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Password sequence panel if active
        if self.password_recording or self.password_verifying or self.stored_password:
            y_offset = 180
            cv2.rectangle(frame, (10, y_offset), (320, y_offset + 120), (20, 20, 20), -1)
            
            # Mode display
            mode_text = "Recording Password" if self.password_recording else "Verifying Password" if self.password_verifying else "Password System"
            cv2.putText(frame, mode_text, (20, y_offset + 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Current sequence
            if self.password_recording or self.password_verifying:
                sequence_str = ''.join(self.blink_sequence)
                remaining_time = max(0, self.SEQUENCE_TIMEOUT - (time.time() - self.sequence_start_time)) if self.sequence_start_time else 0
                
                cv2.putText(frame, f"Sequence: {sequence_str}", (20, y_offset + 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Time left: {remaining_time:.1f}s", (20, y_offset + 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Stored password
            if self.stored_password:
                stored_str = ''.join(self.stored_password)
                cv2.putText(frame, f"Stored pass: {stored_str}", (20, y_offset + 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 255), 2)
        
        # Verification result
        if self.password_result:
            result_color = (0, 255, 0) if "Granted" in self.password_result else (0, 0, 255)
            cv2.putText(frame, self.password_result, (frame.shape[1] - 300, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
        
        # Position indicator (visualize current head position)
        indicator_center = (width - 100, 100)
        indicator_radius = 60
        cv2.circle(frame, indicator_center, indicator_radius, (100, 100, 100), 2)
        
        # Draw axes
        cv2.line(frame, (indicator_center[0], indicator_center[1] - indicator_radius), 
                (indicator_center[0], indicator_center[1] + indicator_radius), (100, 100, 100), 1)
        cv2.line(frame, (indicator_center[0] - indicator_radius, indicator_center[1]), 
                (indicator_center[0] + indicator_radius, indicator_center[1]), (100, 100, 100), 1)
        
        # Draw current position dot
        if not self.metrics['in_dead_zone']:
            # Scale to fit within radius
            vert_scale = min(max(-self.metrics['vertical_displacement'] / (self.THRESHOLDS['vertical'] * 1.5), -1), 1)
            horiz_scale = min(max(self.metrics['horizontal_displacement'] / (self.THRESHOLDS['horizontal'] * 1.5), -1), 1)
            
            dot_x = int(indicator_center[0] + horiz_scale * indicator_radius * 0.8)
            dot_y = int(indicator_center[1] + vert_scale * indicator_radius * 0.8)
            
            cv2.circle(frame, (dot_x, dot_y), 8, (0, 255, 0), -1)
        else:
            # In dead zone
            cv2.circle(frame, indicator_center, 8, (100, 100, 100), -1)
        
        # Instructions
        cv2.rectangle(frame, (10, frame.shape[0] - 100), (320, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(frame, "Controls:", (20, frame.shape[0] - 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "r - reset/recalibrate", (20, frame.shape[0] - 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "p - record password sequence", (20, frame.shape[0] - 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "v - verify password sequence", (20, frame.shape[0] - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 120, height - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def set_password(self, max_time=None, single_window=False):
        """Record a password sequence using neck gestures"""
        print("Starting password recording with neck gestures...")
        self.reset_state()
        
        # Set up camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create a single window for the entire process if requested
        if self.display_ui and single_window:
            cv2.namedWindow("Neck Movement Authentication")
        
        # Set password recording mode
        self.password_recording = True
        start_time = time.time()
        
        # First do calibration
        print("Calibrating... Keep your head in neutral position and look at the camera")
        while not self.calibration['is_complete']:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                cap.release()
                cv2.destroyAllWindows()
                return None
            
            # Process frame
            frame, _ = self.process_frame(frame)
            
            # Display
            if self.display_ui:
                # Create darker background overlay for better text visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
                alpha = 0.6  # Transparency 
                frame[0:150, 0:frame.shape[1]] = cv2.addWeighted(overlay[0:150, 0:frame.shape[1]], 
                                                                alpha, 
                                                                frame[0:150, 0:frame.shape[1]], 
                                                                1 - alpha, 0)
                
                # Add text for calibration state
                cv2.putText(frame, "CALIBRATING", (frame.shape[1]//2 - 100, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, "Keep your head in neutral position", (frame.shape[1]//2 - 200, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Look straight at the camera", (frame.shape[1]//2 - 150, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display calibration progress
                progress = min(100, int((self.calibration['frames'] / 30) * 100))
                progress_width = int((frame.shape[1] - 100) * (progress / 100))
                cv2.rectangle(frame, (50, frame.shape[0] - 50), (frame.shape[1] - 50, frame.shape[0] - 30), (100, 100, 100), -1)
                cv2.rectangle(frame, (50, frame.shape[0] - 50), (50 + progress_width, frame.shape[0] - 30), (0, 255, 255), -1)
                cv2.putText(frame, f"Calibrating: {progress}%", (50, frame.shape[0] - 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if single_window:
                    cv2.imshow("Neck Movement Authentication", frame)
                else:
                    cv2.imshow("Neck Gesture Password System - Calibration", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
            
            # Check for timeout
            if max_time and time.time() - start_time > max_time:
                print("Calibration timeout")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                return None
        
        # Add countdown from 5 to 1 when using single window
        if single_window:
            countdown_start = time.time()
            countdown_duration = 5  # 5 seconds countdown
            
            while time.time() - countdown_start < countdown_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Calculate remaining seconds
                remaining = countdown_duration - int(time.time() - countdown_start)
                
                # Create a clean frame for visualization
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.pose.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                # Convert back to BGR
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Add simple centered countdown with dark background
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                alpha = 0.6  # Transparency
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                # Draw large countdown number
                font_scale = 5.0
                text_thickness = 5
                text = str(remaining)
                
                # Get text size to center it
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                
                # Center position
                text_x = (frame.shape[1] - text_width) // 2
                text_y = (frame.shape[0] + text_height) // 2
                
                # Draw countdown number with shadow for better visibility
                cv2.putText(frame, text, 
                          (text_x + 5, text_y + 5),  # Shadow offset
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2)
                
                cv2.putText(frame, text, 
                          (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
                
                if self.display_ui:
                    cv2.imshow("Neck Movement Authentication", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return None
        
        print("Calibration complete! Now record your password by moving your head:")
        print("Use Up (U), Down (D), Left (L), or Right (R) movements")
        
        # Now record the password with improved UI
        recording_start_time = time.time()
        
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
                    return self.stored_password
                else:
                    cap.release()
                    if self.display_ui:
                        cv2.destroyAllWindows()
                    return None
            
            # Process frame
            frame, gesture = self.process_frame(frame)
            
            # Add recording UI elements
            if self.display_ui:
                # Add recording indicator
                recording_time = current_time - recording_start_time
                cv2.rectangle(frame, (frame.shape[1] - 220, 10), (frame.shape[1] - 10, 50), (0, 0, 0), -1)
                
                # Blinking red dot for recording
                if int(recording_time * 2) % 2 == 0:  # Blink every 0.5 seconds
                    cv2.circle(frame, (frame.shape[1] - 200, 30), 10, (0, 0, 255), -1)
                
                cv2.putText(frame, "RECORDING PATTERN", (frame.shape[1] - 180, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add instruction overlay at the bottom
                cv2.rectangle(frame, (0, frame.shape[0] - 100), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(frame, "Move your head: UP, DOWN, LEFT, or RIGHT to create a sequence", 
                          (20, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, "Current sequence: " + ''.join(self.blink_sequence), 
                          (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                if single_window:
                    cv2.imshow("Neck Movement Authentication", frame)
                else:
                    cv2.imshow("Neck Gesture Password System - Recording", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_state()
                    self.password_recording = True
                    print("Resetting calibration...")
            
            # If password recording is complete, return the password
            if not self.password_recording and self.stored_password:
                password_sequence = ''.join(self.stored_password)
                
                # Show success screen
                if self.display_ui:
                    success_start = time.time()
                    while time.time() - success_start < 3:  # Show for 3 seconds
                        ret, success_frame = cap.read()
                        if not ret:
                            break
                        
                        # Create a dark overlay
                        overlay = success_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (success_frame.shape[1], success_frame.shape[0]), (0, 0, 0), -1)
                        success_frame = cv2.addWeighted(overlay, 0.7, success_frame, 0.3, 0)
                        
                        # Show success message
                        cv2.putText(success_frame, "Pattern Recorded Successfully!", 
                                  (success_frame.shape[1]//2 - 250, success_frame.shape[0]//2 - 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        
                        # Show the recorded pattern
                        cv2.putText(success_frame, f"Your pattern: {password_sequence}",
                                  (success_frame.shape[1]//2 - 150, success_frame.shape[0]//2 + 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        
                        # Display direction icons
                        icon_y = success_frame.shape[0]//2 + 100
                        icon_x_start = success_frame.shape[1]//2 - (len(password_sequence) * 50)
                        
                        for i, char in enumerate(password_sequence):
                            icon_x = icon_x_start + (i * 100)
                            direction = ""
                            
                            if char == 'U':
                                # Draw up arrow
                                cv2.arrowedLine(success_frame, (icon_x, icon_y + 20), (icon_x, icon_y - 20), 
                                              (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
                                direction = "Up"
                            elif char == 'D':
                                # Draw down arrow
                                cv2.arrowedLine(success_frame, (icon_x, icon_y - 20), (icon_x, icon_y + 20), 
                                              (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
                                direction = "Down"
                            elif char == 'L':
                                # Draw left arrow
                                cv2.arrowedLine(success_frame, (icon_x + 20, icon_y), (icon_x - 20, icon_y), 
                                              (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
                                direction = "Left"
                            elif char == 'R':
                                # Draw right arrow
                                cv2.arrowedLine(success_frame, (icon_x - 20, icon_y), (icon_x + 20, icon_y), 
                                              (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.3)
                                direction = "Right"
                            
                            # Add direction text
                            cv2.putText(success_frame, direction, (icon_x - 20, icon_y + 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        cv2.imshow("Neck Movement Authentication", success_frame)
                        cv2.waitKey(1)
                
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                print(f"Password recorded: {password_sequence}")
                return self.stored_password
        
        # Cleanup
        cap.release()
        if self.display_ui:
            cv2.destroyAllWindows()
        
        # Return the password if it was recorded
        if self.stored_password:
            return self.stored_password
        return None
    
    def verify_password(self, stored_password, max_time=None, single_window=False):
        """Verify a password sequence using neck gestures"""
        print("Starting password verification with neck gestures...")
        self.reset_state()
        
        # Convert stored_password to a list if it's a string
        if isinstance(stored_password, str):
            self.stored_password = list(stored_password)
        else:
            self.stored_password = stored_password
        
        print(f"Verifying against pattern: {''.join(self.stored_password)}")
        
        # Set up camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create a single window for the entire process if requested
        if self.display_ui and single_window:
            cv2.namedWindow("Neck Movement Authentication")
        
        start_time = time.time()
        
        # First do calibration
        print("Calibrating... Keep your head in neutral position and look at the camera")
        while not self.calibration['is_complete']:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            # Process frame
            frame, _ = self.process_frame(frame)
            
            # Display
            if self.display_ui:
                # Add text for calibration state
                cv2.putText(frame, "Calibrating... Keep head in neutral position", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if single_window:
                    cv2.imshow("Neck Movement Authentication", frame)
                else:
                    cv2.imshow("Neck Gesture Password System - Calibration", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            # Check for timeout
            if max_time and time.time() - start_time > max_time:
                print("Calibration timeout")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                return False
        
        # Add countdown from 5 to 1
        if single_window:
            countdown_start = time.time()
            countdown_duration = 5  # Change from 3 to 5 seconds
            
            while time.time() - countdown_start < countdown_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Calculate remaining seconds
                remaining = countdown_duration - int(time.time() - countdown_start)
                
                # Create a clean frame for visualization (don't process with metrics)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.pose.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                # Convert back to BGR
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Add simple centered countdown
                # Create background overlay for better visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                alpha = 0.6  # Transparency
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                # Draw large countdown number
                font_scale = 5.0
                text_thickness = 5
                text = str(remaining)
                
                # Get text size to center it
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                
                # Center position
                text_x = (frame.shape[1] - text_width) // 2
                text_y = (frame.shape[0] + text_height) // 2
                
                # Draw countdown number with shadow for better visibility
                cv2.putText(frame, text, 
                          (text_x + 5, text_y + 5),  # Shadow offset
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2)
                
                cv2.putText(frame, text, 
                          (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
                
                # Add instruction text only near the end (last 2 seconds)
                if remaining <= 2:
                    instruction_text = "Get ready to perform your pattern"
                    instruction_font_scale = 1.0
                    
                    # Get text size to center it
                    (instr_width, instr_height), _ = cv2.getTextSize(
                        instruction_text, cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, 2)
                    
                    # Position below the countdown number
                    instr_x = (frame.shape[1] - instr_width) // 2
                    instr_y = text_y + text_height + 50
                    
                    cv2.putText(frame, instruction_text, 
                              (instr_x, instr_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, (0, 255, 255), 2)
                
                if self.display_ui:
                    cv2.imshow("Neck Movement Authentication", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return False
        
        print("Calibration complete! Now enter your password by moving your head:")
        print(f"Enter a {len(stored_password)}-gesture sequence")
        
        # Start verification
        self.password_verifying = True
        
        # Verification loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            current_time = time.time()
            
            # Check for timeout
            if max_time and current_time - start_time > max_time:
                print("Password verification timeout")
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                return False
            
            # Process frame
            frame, gesture = self.process_frame(frame)
            
            # Display
            if self.display_ui:
                if single_window:
                    cv2.imshow("Neck Movement Authentication", frame)
                else:
                    cv2.imshow("Neck Gesture Password System - Verification", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                elif key == ord('r'):
                    self.reset_state()
                    self.password_verifying = True
                    self.stored_password = stored_password
                    print("Resetting calibration...")
            
            # Check if verification is complete
            if not self.password_verifying:
                verification_successful = "Granted" in self.password_result
                
                # If using single window, show result for a moment before closing
                if single_window and self.display_ui:
                    result_start = time.time()
                    result_duration = 3  # Show result for 3 seconds
                    
                    while time.time() - result_start < result_duration:
                        ret, result_frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame for visualization
                        result_frame, _ = self.process_frame(result_frame)
                        
                        # Add result text
                        result_color = (0, 255, 0) if verification_successful else (0, 0, 255)
                        cv2.putText(result_frame, 
                                  "ACCESS GRANTED!" if verification_successful else "ACCESS DENIED!", 
                                  (result_frame.shape[1]//2 - 200, result_frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3)
                        
                        cv2.imshow("Neck Movement Authentication", result_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                
                cap.release()
                if self.display_ui:
                    cv2.destroyAllWindows()
                return verification_successful
            
        # Cleanup
        cap.release()
        if self.display_ui:
            cv2.destroyAllWindows()
        
        # Verification failed if we get here
        return False
    
    def record_password_headless(self, max_time=30, display_ui=None):
        """
        Record a gesture password without requiring UI interaction, suitable for API/frontend integration.
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
                    "password_string": ''.join(password)
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
        Verify a gesture password without requiring UI interaction, suitable for API/frontend integration.
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

def authenticate_user(stored_password=None, max_time=30, display_ui=True, single_window=False):
    """
    Authenticate a user using neck gesture recognition.
    If stored_password is None, it will record a new password.
    If stored_password is provided, it will verify against that password.
    Args:
        stored_password: The password to verify against, or None to record a new password
        max_time: Maximum time to wait in seconds
        display_ui: Whether to display the UI
        single_window: Whether to use a single window for the whole process
    Returns:
        dict: A result dictionary with authentication information
    """
    detector = NeckGestureDetector(display_ui=display_ui)
    
    if stored_password is None:
        # Record a new password
        password = detector.set_password(max_time=max_time, single_window=single_window)
        if password:
            return {
                "success": True,
                "message": "Password successfully recorded",
                "password": password,
                "password_string": ''.join(password)
            }
        else:
            return {
                "success": False,
                "message": "Failed to record password",
                "password": None
            }
    else:
        # Verify existing password
        verification_result = detector.verify_password(stored_password, max_time=max_time, single_window=single_window)
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

def main():
    """Main function for demo and command-line usage"""
    parser = argparse.ArgumentParser(description="Neck Gesture Authentication System")
    parser.add_argument('--headless', action='store_true', help='Run without UI (for API integration)')
    parser.add_argument('--record', action='store_true', help='Record a new password')
    parser.add_argument('--verify', type=str, help='Verify against the provided password string')
    parser.add_argument('--timeout', type=int, default=30, help='Maximum time to wait in seconds')
    parser.add_argument('--single-window', action='store_true', help='Use a single window for calibration and recording')
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.record:
        # Pass the single_window flag to authenticate_user
        result = authenticate_user(stored_password=None, max_time=args.timeout, 
                                  display_ui=not args.headless, single_window=args.single_window)
        print(result)
    elif args.verify:
        result = authenticate_user(stored_password=args.verify, max_time=args.timeout, 
                                  display_ui=not args.headless, single_window=args.single_window)
        print(result)
    else:
        # Interactive mode
        detector = NeckGestureDetector(display_ui=True)
        
        print("Neck Gesture Authentication System")
        print("1. Record a new password")
        print("2. Verify a password")
        print("3. Live demo mode")
        choice = input("Choose an option: ")
        
        if choice == '1':
            password = detector.set_password()
            if password:
                print(f"Password recorded: {''.join(password)}")
        elif choice == '2':
            password_str = input("Enter the password to verify against: ")
            result = detector.verify_password(list(password_str))
            print(f"Verification result: {'Success' if result else 'Failed'}")
        else:
            # Live demo mode
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                try:
                    frame, gesture = detector.process_frame(frame)
                    if gesture:
                        print(f"Detected gesture: {gesture}")
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Display
                cv2.imshow("Neck Gesture Detection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    detector.reset_state()
                    print("Resetting calibration...")
                elif key == ord('p'):
                    detector.password_recording = True
                    print("Recording password sequence...")
                elif key == ord('v'):
                    if detector.stored_password:
                        detector.password_verifying = True
                        detector.blink_sequence = []
                        detector.sequence_start_time = time.time()
                        print("Verifying password sequence...")
                    else:
                        print("No stored password to verify against")
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
