from architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import tensorflow as tf
import tqdm
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train face recognition model')
parser.add_argument('--face-data', type=str, default='Faces/',
                    help='Directory containing face images')
parser.add_argument('--output', type=str, default='encodings/encodings.pkl',
                    help='Output file for face encodings')
parser.add_argument('--model-weights', type=str, default='facenet_keras_weights.h5',
                    help='Path to model weights')
parser.add_argument('--confidence', type=float, default=0.95,
                    help='Minimum confidence for face detection')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size for encoding')
args = parser.parse_args()

# Paths and variables
face_data = args.face_data
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights(args.model_weights)

# Try different parameter naming depending on MTCNN implementation
try:
    # Option 1: For some implementations
    face_detector = mtcnn.MTCNN(min_face_size=20, scale_factor=0.709)
except TypeError:
    try:
        # Option 2: For different implementations
        face_detector = mtcnn.MTCNN(minsize=20, factor=0.709)
    except TypeError:
        # Fallback with no parameters
        print("Warning: Using default MTCNN parameters - check documentation for your specific MTCNN version")
        face_detector = mtcnn.MTCNN()

print("Initializing face detector...")
face_detector = mtcnn.MTCNN()

encoding_dict = dict()
l2_normalizer = Normalizer('l2')

# Make sure output directory exists
Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)

def normalize(img):
    """Normalize the pixel values of the image."""
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def align_face(img, landmarks):
    """Align the face based on eye positions."""
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # Calculate the center of each eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    
    # Calculate angle between eyes
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Desired eye position
    desired_left_eye = (0.35, 0.35)
    desired_right_eye_x = 1.0 - desired_left_eye[0]
    
    # Calculate scale and get rotation matrix
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0]) * required_shape[0]
    scale = desired_dist / dist
    
    # Calculate center point between eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    
    # Update the translation component of the matrix
    tX = required_shape[0] * 0.5
    tY = required_shape[1] * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    
    # Apply the affine transformation
    output = cv2.warpAffine(img, M, required_shape, flags=cv2.INTER_CUBIC)
    return output

def get_face_landmarks(detector_result):
    """Extract facial landmarks from MTCNN detection."""
    landmarks = {}
    keypoints = detector_result['keypoints']
    landmarks['left_eye'] = [keypoints['left_eye']]
    landmarks['right_eye'] = [keypoints['right_eye']]
    landmarks['nose'] = [keypoints['nose']]
    landmarks['mouth_left'] = [keypoints['mouth_left']]
    landmarks['mouth_right'] = [keypoints['mouth_right']]
    return landmarks

def augment_face(face):
    """Apply simple augmentations to the face image."""
    augmented = []
    # Original
    augmented.append(face)
    
    # Slightly lighter
    light = cv2.convertScaleAbs(face, alpha=1.1, beta=10)
    augmented.append(light)
    
    # Slightly darker
    dark = cv2.convertScaleAbs(face, alpha=0.9, beta=-10)
    augmented.append(dark)
    
    # Horizontal flip (mirror)
    flipped = cv2.flip(face, 1)  # 1 for horizontal flip
    augmented.append(flipped)
    
    return augmented

def process_faces_in_batches(face_list, batch_size=32):
    """Process faces in batches for more efficient encoding."""
    encodings = []
    
    # Create batches
    for i in range(0, len(face_list), batch_size):
        batch = np.array(face_list[i:i+batch_size])
        # Get encodings for batch
        batch_encodings = face_encoder.predict(batch)
        encodings.extend(batch_encodings)
    
    return encodings

# Main training loop with progress bar
print("Starting face encoding process...")
all_persons = list(os.listdir(face_data))
all_encodings = {}

for person_name in tqdm.tqdm(all_persons, desc="Processing people"):
    person_dir = os.path.join(face_data, person_name)
    if not os.path.isdir(person_dir):
        continue
    
    face_images = []
    person_images = list(os.listdir(person_dir))
    
    for image_name in tqdm.tqdm(person_images, desc=f"Processing {person_name}", leave=False):
        image_path = os.path.join(person_dir, image_name)
        
        try:
            # Load and convert image
            img_BGR = cv2.imread(image_path)
            if img_BGR is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detection_results = face_detector.detect_faces(img_RGB)
            
            # Skip if no face detected or confidence too low
            if not detection_results or detection_results[0]['confidence'] < args.confidence:
                print(f"Warning: No confident face detection in {image_path}")
                continue
            
            # Get the main face
            detection = detection_results[0]
            x1, y1, width, height = detection['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1+width, y1+height
            
            # Extract face with a margin for better alignment
            margin = int(0.3 * width)  # 30% margin
            face = img_RGB[max(0, y1-margin):min(img_RGB.shape[0], y2+margin), 
                          max(0, x1-margin):min(img_RGB.shape[1], x2+margin)]
            
            # Get facial landmarks and align face
            landmarks = get_face_landmarks(detection)
            try:
                face = align_face(face, landmarks)
            except Exception as e:
                # If alignment fails, use standard cropping
                face = cv2.resize(img_RGB[y1:y2, x1:x2], required_shape)
            
            # Normalize and preprocess
            face = normalize(face)
            
            # Apply augmentations
            augmented_faces = augment_face(face)
            
            # Add to list for batch processing
            for aug_face in augmented_faces:
                face_d = np.expand_dims(cv2.resize(aug_face, required_shape), axis=0)
                face_images.append(face_d[0])
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Process all faces for this person in batches if we have any
    if face_images:
        print(f"Encoding {len(face_images)} face images for {person_name}...")
        person_encodings = process_faces_in_batches(face_images, args.batch_size)
        
        # Average all encodings for this person
        if person_encodings:
            final_encoding = np.mean(person_encodings, axis=0)
            final_encoding = l2_normalizer.transform(np.expand_dims(final_encoding, axis=0))[0]
            encoding_dict[person_name] = final_encoding
            print(f"Successfully encoded {person_name}")
        else:
            print(f"Warning: No valid encodings generated for {person_name}")
    else:
        print(f"Warning: No valid face images found for {person_name}")

# Save encodings
print(f"Saving {len(encoding_dict)} face encodings to {args.output}...")
with open(args.output, 'wb') as file:
    pickle.dump(encoding_dict, file)

print("Face encoding process completed!")






