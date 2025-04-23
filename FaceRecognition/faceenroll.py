import cv2
import mediapipe as mp
import numpy as np
import os
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FaceDataCollector:
    """Class for collecting facial data with visual guidance"""
    
    # Define the poses to capture
    CAPTURE_POSES = ['front', 'left', 'right', 'up', 'down']
    
    def __init__(self, output_dir="Faces", display_ui=True):
        """Initialize the face data collector"""
        self.output_dir = output_dir
        self.display_ui = display_ui
        self.quality_threshold = 0.65
        
        # Initialize state variables
        self.reset_state()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create Face Mesh with appropriate settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logging.info("MediaPipe initialized successfully")
        
        # Initialize performance metrics
        self.process_times = []
        self.fps = 0
        
        # Direction timing
        self.direction_change_time = time.time()
    
    def reset_state(self):
        """Reset the internal state"""
        self.session_active = False
        self.current_person = None
        self.current_pose_index = 0
        self.current_face_landmarks = None
        self.face_detected = False
        self.target_pose = [0, 0, 0]  # Target yaw, pitch, roll
        self.collected_frames = 0
        self.captured_frames = []
    
    def start_session(self, person_name):
        """Start a data collection session for a person"""
        self.session_active = True
        self.current_person = person_name
        self.current_pose_index = 0
        self.collected_frames = 0
        self.captured_frames = []
        
        # Create directory for this person
        person_dir = os.path.join(self.output_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            logging.info(f"Created directory for {person_name}: {person_dir}")
        
        logging.info(f"Starting data collection session for: {person_name}")
    
    def check_face_pose(self, frame, face_mesh_results):
        """Check if the face is in the correct pose"""
        if not face_mesh_results or not face_mesh_results.multi_face_landmarks:
            self.face_detected = False
            return False, [0, 0, 0]
        
        self.face_detected = True
        landmarks = face_mesh_results.multi_face_landmarks[0].landmark
        self.current_face_landmarks = landmarks
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Extract face orientation (simplified)
        # For accurate face pose estimation, additional math is needed
        # This is a simplified approximation
        orientation = [0, 0, 0]  # Yaw, pitch, roll
        
        # Simplified - just checking if face is roughly frontal
        nose_tip = landmarks[4]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
        left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
        right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
        
        # Calculate approximate yaw (left-right rotation)
        eye_dist = abs(right_eye_x - left_eye_x)
        nose_offset = nose_x - (left_eye_x + right_eye_x) / 2
        orientation[0] = (nose_offset / eye_dist) * 45  # Rough approximation of yaw angle
        
        # Calculate approximate pitch (up-down rotation)
        eye_y = (left_eye_y + right_eye_y) / 2
        nose_y_offset = nose_y - eye_y
        orientation[1] = (nose_y_offset / eye_dist) * 45  # Rough approximation of pitch angle
        
        # Roll calculation (simplified)
        eyes_slope = (right_eye_y - left_eye_y) / max(1, right_eye_x - left_eye_x)
        orientation[2] = np.arctan(eyes_slope) * 180 / np.pi  # Roll in degrees
        
        # Determine if the pose is correct (simplified)
        yaw_ok = abs(orientation[0] - self.target_pose[0]) < 15
        pitch_ok = abs(orientation[1] - self.target_pose[1]) < 15
        roll_ok = abs(orientation[2] - self.target_pose[2]) < 15
        
        return (yaw_ok and pitch_ok and roll_ok), orientation
    
    def assess_image_quality(self, frame):
        """Assess the quality of the face image"""
        # Skip if no face is detected
        if not self.face_detected or self.current_face_landmarks is None:
            return 0.0, {"face_size": 0.0, "blur": 1.0}
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Calculate face size based on landmarks
        landmarks = self.current_face_landmarks
        
        # Use landmarks to find face bounding box
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Convert to pixel coordinates
        min_x, max_x = int(min_x * w), int(max_x * w)
        min_y, max_y = int(min_y * h), int(max_y * h)
        
        # Calculate face size relative to frame
        face_width = max_x - min_x
        face_height = max_y - min_y
        face_size_ratio = (face_width * face_height) / (w * h)
        
        # Detect blur using Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[max(0, min_y):min(h, max_y), max(0, min_x):min(w, max_x)]
        
        if face_roi.size == 0:  # Empty ROI check
            blur_score = 0
        else:
            blur = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            # Normalize blur (higher is better)
            blur_score = min(1.0, blur / 500)  
        
        # Calculate overall quality score
        # Weighted combination of factors
        size_score = min(1.0, face_size_ratio * 30)  # Adjust multiplier as needed
        
        metrics = {
            "face_size": size_score,
            "blur": blur_score
        }
        
        # Final quality is weighted average
        quality = 0.6 * size_score + 0.4 * blur_score
        
        return quality, metrics
    
    def extract_face_and_save(self, frame, landmarks, margin=0.2, quality_threshold=0.5):
        """Extract face region and save to disk"""
        if not self.session_active or landmarks is None:
            return False, None
        
        # Check quality first
        quality, _ = self.assess_image_quality(frame)
        if quality < quality_threshold:
            logging.debug(f"Skipping low quality face: {quality:.2f} < {quality_threshold}")
            return False, None
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Get face bounding box
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add margin
        width = max_x - min_x
        height = max_y - min_y
        min_x = max(0, min_x - width * margin)
        max_x = min(1, max_x + width * margin)
        min_y = max(0, min_y - height * margin)
        max_y = min(1, max_y + height * margin)
        
        # Convert to pixel coordinates
        min_x, max_x = int(min_x * w), int(max_x * w)
        min_y, max_y = int(min_y * h), int(max_y * h)
        
        # Extract face region
        face_img = frame[min_y:max_y, min_x:max_x]
        
        # Skip if extraction failed
        if face_img.size == 0:
            logging.warning("Extracted face has no size")
            return False, None
        
        # Create filename with timestamp and pose
        pose_name = self.CAPTURE_POSES[self.current_pose_index] if self.current_pose_index < len(self.CAPTURE_POSES) else "extra"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.current_person}_{pose_name}_{timestamp}.jpg"
        file_path = os.path.join(self.output_dir, self.current_person, filename)
        
        # Save image
        cv2.imwrite(file_path, face_img)
        self.collected_frames += 1
        self.captured_frames.append(file_path)
        
        return True, file_path
    
    def collect_face_data_with_bouncing_guides(self, person_name=None, max_time=None):
        """Collect face data with timer-based capture and bouncing visual guides"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return {"success": False, "message": "Failed to open camera"}
        
        # Start a session if person name is provided
        if person_name:
            self.start_session(person_name)
        
        # For timing
        start_time = time.time()
        last_capture_time = 0
        capture_interval = 1  # Capture every second
        
        # Direction guidance variables
        directions = ['center', 'left', 'right', 'up', 'down']
        current_direction_index = 0
        self.direction_change_time = start_time
        direction_interval = 3  # Change direction every 3 seconds
        
        # Get frame dimensions for UI
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
        else:
            height, width = 720, 1280  # Fallback
        
        # Main loop
        try:
            while True:
                # Check for timeout
                current_time = time.time()
                if max_time and current_time - start_time > max_time:
                    logging.info(f"Maximum collection time of {max_time}s reached")
                    break
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to grab frame")
                    continue
                
                # Process frame with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                face_mesh_results = self.face_mesh.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                # Check face pose (just to update face detection status)
                self.check_face_pose(frame, face_mesh_results)
                
                # Change direction at intervals
                if (current_time - self.direction_change_time) > direction_interval:
                    current_direction_index = (current_direction_index + 1) % len(directions)
                    self.direction_change_time = current_time
                    logging.info(f"Changing direction to: {directions[current_direction_index]}")
                
                # Direct auto-capture every second regardless of face quality
                if self.session_active and (current_time - last_capture_time) > capture_interval:
                    logging.info("Capturing frame every second...")
                    success, file_path = self.extract_face_and_save(frame, self.current_face_landmarks, quality_threshold=0.3)
                    if success:
                        logging.info(f"Successfully captured frame: {file_path}")
                        last_capture_time = current_time
                    else:
                        logging.warning("Automatic capture failed")
                
                # Draw bouncing visual cue for current direction
                current_direction = directions[current_direction_index]
                result_frame = self.draw_bouncing_visual_cue(frame, current_direction)
                
                # Display the result
                if self.display_ui:
                    cv2.imshow("Face Data Collection", result_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logging.info("User requested to quit")
                        break
                    elif key == ord('r'):
                        logging.info("User requested to reset")
                        self.reset_state()
                    elif key == ord('s') and not self.session_active:
                        # Prompt for name
                        person_name = input("Enter person's name: ")
                        if person_name:
                            self.start_session(person_name)
                    elif key == ord('c') and self.session_active:
                        # Manual capture
                        success, file_path = self.extract_face_and_save(
                            frame, self.current_face_landmarks, quality_threshold=0.3)
                        if success:
                            logging.info(f"Manually captured: {file_path}")
                
                # Check if we've collected enough photos
                if self.session_active and self.collected_frames >= 30:  # Adjust number as needed
                    logging.info("Collected enough frames. Collection complete!")
                    break
                    
        except Exception as e:
            logging.error(f"Error during face data collection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release resources
            cap.release()
            if self.display_ui:
                cv2.destroyAllWindows()
        
        # Prepare results
        results = {
            "success": True,
            "person": self.current_person,
            "frames_collected": self.collected_frames,
            "captured_frames": self.captured_frames,
            "elapsed_time": time.time() - start_time
        }
        
        return results

    def draw_bouncing_visual_cue(self, frame, direction):
        """Draw a bouncing visual cue for face movement directions with clear instructions"""
        height, width = frame.shape[:2]
        result_frame = frame.copy()
        
        # Define the position and size of the bouncing cue
        cue_size = 50
        center_x = width // 2
        center_y = height // 2
        cue_x = center_x
        cue_y = center_y
        
        # Calculate offset for the bouncing cue
        offset = 150  # Distance from center
        
        # Position the cue based on direction
        if direction == 'left':
            cue_x = center_x - offset
            instruction = "Look LEFT"
        elif direction == 'right':
            cue_x = center_x + offset
            instruction = "Look RIGHT"
        elif direction == 'up':
            cue_y = center_y - offset
            instruction = "Look UP"
        elif direction == 'down':
            cue_y = center_y + offset
            instruction = "Look DOWN"
        else:  # center
            cue_x = center_x
            cue_y = center_y
            instruction = "Look at CENTER"
        
        # Draw the cue
        cv2.circle(result_frame, (cue_x, cue_y), cue_size, (0, 165, 255), -1)  # Orange filled circle
        
        # Draw text instructions inside the circle
        text_size = cv2.getTextSize("LOOK", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = cue_x - text_size[0] // 2
        text_y = cue_y
        cv2.putText(result_frame, "LOOK", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw direction text below the circle
        cv2.putText(result_frame, instruction, (cue_x - 70, cue_y + cue_size + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw a progress indicator for current direction
        time_in_current_direction = time.time() - self.direction_change_time
        progress = min(time_in_current_direction / 3.0, 1.0)  # 3 seconds per direction
        bar_width = 200
        filled_width = int(bar_width * progress)
        
        cv2.rectangle(result_frame, (width//2 - bar_width//2, height - 50), 
                     (width//2 + bar_width//2, height - 30), (100, 100, 100), -1)
        cv2.rectangle(result_frame, (width//2 - bar_width//2, height - 50),
                     (width//2 - bar_width//2 + filled_width, height - 30), (0, 255, 0), -1)
        
        # Display capture status
        cv2.putText(result_frame, f"Auto-capturing every second | Total: {self.collected_frames}", 
                   (width//2 - 150, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw a dashed line from center to the cue
        if direction != 'center':
            for i in range(0, offset, 20):
                if direction == 'left':
                    pt1 = (center_x - i, center_y)
                    pt2 = (center_x - i - 10, center_y)
                elif direction == 'right':
                    pt1 = (center_x + i, center_y)
                    pt2 = (center_x + i + 10, center_y)
                elif direction == 'up':
                    pt1 = (center_x, center_y - i)
                    pt2 = (center_x, center_y - i - 10)
                elif direction == 'down':
                    pt1 = (center_x, center_y + i)
                    pt2 = (center_x, center_y + i + 10)
                
                cv2.line(result_frame, pt1, pt2, (0, 255, 255), 2)
        
        # If face is detected, show it with a rectangle
        if self.face_detected and self.current_face_landmarks:
            h, w = frame.shape[:2]
            landmarks = self.current_face_landmarks
            
            # Get face bounding box
            x_coords = [landmark.x for landmark in landmarks]
            y_coords = [landmark.y for landmark in landmarks]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Add margin
            width_face = max_x - min_x
            height_face = max_y - min_y
            margin = 0.1
            min_x = max(0, min_x - width_face * margin)
            max_x = min(1, max_x + width_face * margin)
            min_y = max(0, min_y - height_face * margin)
            max_y = min(1, max_y + height_face * margin)
            
            # Convert to pixel coordinates
            min_x, max_x = int(min_x * w), int(max_x * w)
            min_y, max_y = int(min_y * h), int(max_y * h)
            
            # Draw rectangle around face
            cv2.rectangle(result_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        
        return result_frame


def collect_faces_headless(person_name, output_dir="Faces", max_time=60, display_ui=False):
    """
    Collect face data without requiring UI interaction, suitable for API/frontend integration.
    
    Args:
        person_name: Name of the person whose face is being collected
        output_dir: Directory to save face images
        max_time: Maximum time for collection in seconds
        display_ui: Whether to display the UI
        
    Returns:
        dict: A result dictionary with collection status information
    """
    collector = FaceDataCollector(output_dir=output_dir, display_ui=display_ui)
    
    try:
        results = collector.collect_face_data_with_bouncing_guides(person_name=person_name, max_time=max_time)
        
        if results["frames_collected"] > 0:
            return {
                "success": True,
                "message": f"Successfully collected {results['frames_collected']} face images",
                "person": results["person"],
                "frames_collected": results["frames_collected"],
                "captured_frames": results.get("captured_frames", []),
                "elapsed_time": results.get("elapsed_time", 0)
            }
        else:
            return {
                "success": False,
                "message": "Failed to collect any face images",
                "person": person_name,
                "frames_collected": 0
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error during face collection: {str(e)}",
            "person": person_name,
            "frames_collected": 0
        }


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Data Collection System")
    parser.add_argument('--headless', action='store_true', help='Run without UI (for API integration)')
    parser.add_argument('--name', type=str, help='Name of the person to collect face data for')
    parser.add_argument('--output', type=str, default="Faces", help='Output directory for face images')
    parser.add_argument('--timeout', type=int, default=60, help='Maximum time to collect faces in seconds')
    
    args = parser.parse_args()
    
    # If name is provided via command line, use it directly
    if args.name:
        result = collect_faces_headless(
            person_name=args.name, 
            output_dir=args.output, 
            max_time=args.timeout, 
            display_ui=not args.headless
        )
        print(result)
    else:
        # Interactive mode
        collector = FaceDataCollector(output_dir=args.output)
        
        # Ask for person's name
        person_name = input("Enter person's name: ")
        
        # Run collection with timeout
        results = collector.collect_face_data_with_bouncing_guides(person_name=person_name, max_time=args.timeout)
        
        # Display results
        print("\nCollection Results:")
        print(f"Person: {results['person']}")
        print(f"Frames collected: {results['frames_collected']}")
        print(f"Collection complete: {results.get('collection_complete', True)}")
