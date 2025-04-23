import os
import json
import uuid
import platform
import threading
import time
import subprocess
from werkzeug.utils import secure_filename
import shutil
import shutil
import pickle
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, Response, stream_with_context, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import sys
# Add the EyeBlink directory to the path
sys.path.append(os.path.join(os.getcwd(), 'EyeBlink'))
import EBR

import re
import copy
import base64
import io
import threading
import time
import cv2
import numpy as np

# Add path to Voice directory
sys.path.append(os.path.join(os.getcwd(), 'Voice'))
from Voice import voicerecord, voicevalidate

# Import EBR module directly
from EyeBlink.EBR import EyeBlinkAuthenticator, processing_active, global_frame

# Add this import to ensure we can directly use the EBR module
sys.path.insert(0, os.path.join(os.getcwd(), 'EyeBlink'))

# Add these imports
import io
import queue
import sys
import threading
import requests
import subprocess
import time
import signal
from threading import Thread

# Add these imports at the top of app.py
from cryptography.fernet import Fernet
import base64
import os

app = Flask(__name__, static_folder='static')
app.secret_key = 'supersecretkey'  # Change this for production

# Modify these variables to handle the camera directly in Flask
camera = None
camera_active = False
camera_frame = None
eyeblink_processing = False
eyeblink_result = None

# Add this near the top where other global variables are defined
erb_process = None
erb_thread_active = False

# Variables to track eyeblink authentication status
eyeblink_auth = None
eyeblink_thread_active = False

# Add this global variable to track the RAG app process
rag_app_process = None
RAG_APP_PORT = 5001  # Use a different port than your main app

# Define paths
BASE_DIR = os.getcwd()
DATA_FILE = os.path.join(BASE_DIR, "Biometric", "biometric.json")
PROTECTED_FOLDER = os.path.join(BASE_DIR, "ProtectedFiles")
TEMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
ENCRYPTION_KEY_FILE = os.path.join(BASE_DIR, "crypto.key")

# Ensure the ProtectedFiles folder exists.
if not os.path.exists(PROTECTED_FOLDER):
    os.makedirs(PROTECTED_FOLDER)

# Ensure the temp_uploads folder exists.
if not os.path.exists(TEMP_UPLOAD_FOLDER):
    os.makedirs(TEMP_UPLOAD_FOLDER)

# Initialize biometric.json if it doesn't exist.
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump({"users": [], "files": []}, f, indent=4)

# Function to get or create encryption key
def get_encryption_key():
    """Get the encryption key or generate one if it doesn't exist"""
    if os.path.exists(ENCRYPTION_KEY_FILE):
        with open(ENCRYPTION_KEY_FILE, 'rb') as key_file:
            key = key_file.read()
    else:
        # Generate a new key
        key = Fernet.generate_key()
        # Save the key to file with restrictive permissions
        with open(ENCRYPTION_KEY_FILE, 'wb') as key_file:
            key_file.write(key)
        # Set file permissions to be restrictive (only owner can read)
        try:
            os.chmod(ENCRYPTION_KEY_FILE, 0o600)
        except:
            pass  # Permissions might not work on all platforms
    return key

# Create encryption and decryption functions
def encrypt_data(data):
    """Encrypt string data"""
    if not data:
        return data
    key = get_encryption_key()
    cipher = Fernet(key)
    # Convert to bytes if it's a string
    if isinstance(data, str):
        data_bytes = data.encode()
    else:
        data_bytes = data
    # Encrypt the data
    encrypted_data = cipher.encrypt(data_bytes)
    # Return base64 encoded string for JSON storage
    return base64.b64encode(encrypted_data).decode('utf-8')

def decrypt_data(encrypted_data):
    """Decrypt string data"""
    if not encrypted_data:
        return encrypted_data
    try:
        key = get_encryption_key()
        cipher = Fernet(key)
        # Decode from base64
        encrypted_bytes = base64.b64decode(encrypted_data)
        # Decrypt
        decrypted_data = cipher.decrypt(encrypted_bytes)
        # Return as string
        return decrypted_data.decode('utf-8')
    except Exception as e:
        print(f"Error decrypting data: {str(e)}")
        return None

# Add these functions after the existing encryption functions
def encrypt_file(input_file_path, output_file_path):
    """Encrypt a file using the app's encryption key"""
    key = get_encryption_key()
    cipher = Fernet(key)
    
    try:
        # Read the input file
        with open(input_file_path, 'rb') as file:
            file_data = file.read()
        
        # Encrypt the data
        encrypted_data = cipher.encrypt(file_data)
        
        # Write the encrypted data to the output file
        with open(output_file_path, 'wb') as file:
            file.write(encrypted_data)
        
        return True
    except Exception as e:
        print(f"Error encrypting file: {str(e)}")
        return False

def decrypt_file(encrypted_file_path, output_file_path=None):
    """
    Decrypt a file using the app's encryption key
    If output_file_path is None, returns the decrypted data instead of writing to file
    """
    key = get_encryption_key()
    cipher = Fernet(key)
    
    try:
        # Read the encrypted file
        with open(encrypted_file_path, 'rb') as file:
            encrypted_data = file.read()
        
        # Decrypt the data
        decrypted_data = cipher.decrypt(encrypted_data)
        
        if output_file_path:
            # Write the decrypted data to the output file
            with open(output_file_path, 'wb') as file:
                file.write(decrypted_data)
            return True
        else:
            # Return the decrypted data
            return decrypted_data
    except Exception as e:
        print(f"Error decrypting file: {str(e)}")
        return None

def read_data():
    """Read data from JSON file and decrypt sensitive fields"""
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    
    # Decrypt sensitive fields for each user
    for user in data.get('users', []):
        # Decrypt PIN
        if 'encrypted_pin' in user:
            user['pin'] = decrypt_data(user['encrypted_pin'])
            # Remove encrypted_pin from the in-memory data
            user.pop('encrypted_pin', None)
        
        # Decrypt biometrics
        if 'encrypted_biometrics' in user:
            biometrics_json = decrypt_data(user['encrypted_biometrics'])
            if biometrics_json:
                user['biometrics'] = json.loads(biometrics_json)
            else:
                user['biometrics'] = {}
            # Remove encrypted_biometrics from the in-memory data
            user.pop('encrypted_biometrics', None)
            
        # Decrypt hospital dashboard settings if they exist
        if 'encrypted_hospital_settings' in user:
            hospital_settings_json = decrypt_data(user['encrypted_hospital_settings'])
            if hospital_settings_json:
                user['hospital_settings'] = json.loads(hospital_settings_json)
            else:
                user['hospital_settings'] = {}
            # Remove encrypted_hospital_settings from the in-memory data
            user.pop('encrypted_hospital_settings', None)
    
    return data

def write_data(data):
    """Encrypt sensitive fields and write data to JSON file"""
    # Create a copy of the data to avoid modifying the original
    data_copy = copy.deepcopy(data)
    
    # Encrypt sensitive fields for each user
    for user in data_copy.get('users', []):
        # Encrypt PIN
        if 'pin' in user:
            user['encrypted_pin'] = encrypt_data(user['pin'])
            # Remove unencrypted pin from the data to be written
            user.pop('pin', None)
        
        # Encrypt biometrics
        if 'biometrics' in user:
            biometrics_json = json.dumps(user['biometrics'])
            user['encrypted_biometrics'] = encrypt_data(biometrics_json)
            # Remove unencrypted biometrics from the data to be written
            user.pop('biometrics', None)
            
        # Encrypt hospital dashboard settings
        if 'hospital_settings' in user:
            hospital_settings_json = json.dumps(user['hospital_settings'])
            user['encrypted_hospital_settings'] = encrypt_data(hospital_settings_json)
            # Remove unencrypted hospital_settings from the data to be written
            user.pop('hospital_settings', None)
    
    # Write the encrypted data to file
    with open(DATA_FILE, 'w') as f:
        json.dump(data_copy, f, indent=4)

def get_user_by_username(username):
    data = read_data()
    for user in data['users']:
        if user['username'] == username:
            return user
    return None

def get_user_by_id(user_id):
    data = read_data()
    for user in data['users']:
        if user['id'] == user_id:
            return user
    return None

def update_user(user):
    """Update a user in the database"""
    data = read_data()
    for idx, u in enumerate(data['users']):
        if u['id'] == user['id']:
            # Ensure biometrics are properly handled (decrypted in memory)
            if 'biometrics' in user and 'encrypted_biometrics' in u:
                # We're updating a user with decrypted biometrics
                data['users'][idx] = user
            else:
                # Handle case where user might have encrypted fields
                if 'encrypted_biometrics' in u and 'biometrics' not in user:
                    # Preserve encrypted biometrics if not being updated
                    user['encrypted_biometrics'] = u['encrypted_biometrics']
                
                if 'encrypted_pin' in u and 'pin' not in user:
                    # Preserve encrypted pin if not being updated
                    user['encrypted_pin'] = u['encrypted_pin']
                
                data['users'][idx] = user
            break
    write_data(data)

def get_file_by_id(file_id):
    data = read_data()
    for f in data['files']:
        if f['id'] == file_id:
            return f
    return None

def migrate_to_encrypted_data():
    """Migrate existing plain-text data to encrypted format"""
    try:
        # Read the data file directly without decryption
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        # Check if migration is needed
        migration_needed = False
        for user in data.get('users', []):
            if 'pin' in user and 'encrypted_pin' not in user:
                migration_needed = True
                break
            if 'biometrics' in user and 'encrypted_biometrics' not in user:
                migration_needed = True
                break
        
        if not migration_needed:
            print("Data already encrypted, no migration needed.")
            return
        
        # Create a backup of the data file
        backup_file = f"{DATA_FILE}.backup_{int(time.time())}"
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Backup created: {backup_file}")
        
        # Process the data and encrypt sensitive fields
        for user in data.get('users', []):
            # Encrypt PIN
            if 'pin' in user:
                user['encrypted_pin'] = encrypt_data(user['pin'])
                user.pop('pin', None)
            
            # Encrypt biometrics
            if 'biometrics' in user:
                biometrics_json = json.dumps(user['biometrics'])
                user['encrypted_biometrics'] = encrypt_data(biometrics_json)
                user.pop('biometrics', None)
        
        # Write the encrypted data back to the file
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data migration completed successfully.")
    except Exception as e:
        print(f"Error during data migration: {str(e)}")

# Add this function to clean up temporary files
def cleanup_temp_directories():
    """Clean up temporary directories when the application starts"""
    try:
        # Clean temp_uploads directory
        if os.path.exists(TEMP_UPLOAD_FOLDER):
            for filename in os.listdir(TEMP_UPLOAD_FOLDER):
                file_path = os.path.join(TEMP_UPLOAD_FOLDER, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting temp upload file {file_path}: {e}")
        
        # Clean temp_decrypted directory
        temp_decrypted_dir = os.path.join(BASE_DIR, "temp_decrypted")
        if os.path.exists(temp_decrypted_dir):
            for filename in os.listdir(temp_decrypted_dir):
                file_path = os.path.join(temp_decrypted_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting temp decrypted file {file_path}: {e}")
                    
        print("Temporary directories cleaned up successfully")
    except Exception as e:
        print(f"Error cleaning up temporary directories: {e}")

# Function to start the RAG app
def start_rag_app():
    global rag_app_process
    
    # Check if the process already exists and is running
    if rag_app_process is not None:
        # Check if process is still running
        if rag_app_process.poll() is None:  # None means it's still running
            return
    
    try:
        # Get the path to the RAG app
        rag_app_path = os.path.join(BASE_DIR, "rag", "Frontend_Connection", "app.py")
        
        # Ensure the directory exists
        if not os.path.exists(rag_app_path):
            print(f"RAG app not found at {rag_app_path}")
            return
            
        # Start the RAG app as a separate process that will keep running
        # even if the parent process (this Flask app) exits
        rag_app_process = subprocess.Popen(
            ["python", rag_app_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # This ensures the process continues running independently
            start_new_session=True
        )
        
        print(f"Started RAG app with PID: {rag_app_process.pid}")
    except Exception as e:
        print(f"Error starting RAG app: {str(e)}")

@app.route('/video_feed')
def video_feed():
    def generate():
        # Just return a placeholder image to avoid camera access
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 30
        cv2.putText(placeholder, "EBR window opens separately", (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        while True:
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)
    
    return Response(stream_with_context(generate()),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Update record_biometric for eyeblink and voice
@app.route('/record_biometric', methods=['POST'])
def record_biometric():
    global eyeblink_processing, eyeblink_result
    
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    # Get the biometric type from form data or JSON data depending on how it's sent
    biometric_type = request.form.get('biometric_type')
    
    # If not in form data, try to get from JSON
    if not biometric_type and request.is_json:
        data = request.get_json()
        biometric_type = data.get('biometric_type')
    
    print(f"Recording biometric: {biometric_type}")
    
    if biometric_type == 'fingerprint':
        # Check if image file is included in the request
        if 'fingerprint_image' in request.files:
            fingerprint_file = request.files['fingerprint_image']
            
            # Get sequence information
            sequence_number = int(request.form.get('sequence_number', 1))
            total_prints = int(request.form.get('total_prints', 1))
            
            # Import the enroll module
            from FingerPrint.enroll import save_fingerprint
            
            # Save the fingerprint image
            try:
                user_id = session['user_id']
                fingerprint_path = save_fingerprint(fingerprint_file, user_id, sequence_number, total_prints)
                return jsonify({
                    'success': True,
                    'code': fingerprint_path,
                    'sequence_number': sequence_number,
                    'total_prints': total_prints,
                    'message': f'Fingerprint #{sequence_number} saved successfully'
                })
            except Exception as e:
                print(f"Error saving fingerprint: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        else:
            # Fallback to dummy code if no image is provided
            code = fingerprintsave.save(None)
            return jsonify({'success': True, 'code': code})
    elif biometric_type == 'eyeblink':
        # Don't start if already processing
        if eyeblink_processing:
            return jsonify({'success': True, 'in_progress': True})
        
        # Reset previous result and mark as processing
        eyeblink_result = None
        eyeblink_processing = True
        
        try:
            # Make sure to kill ANY Python processes that might be running EBR.py
            # This is critical to prevent duplicate windows
            os.system("pkill -f 'python.*EBR.py'")
            
            # IMPORTANT: Do not import the EBR modules directly here
            # Use subprocess only to avoid opening two windows
            
            # Run EBR.py with the UI window - this will be the only window that opens
            result = subprocess.run(
                ['python', os.path.join(BASE_DIR, 'EyeBlink', 'EBR.py'), '--record'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Process output
            output = result.stdout.strip()
            print(f"EBR.py output: {output}")
            
            # Parse the password from output
            match = re.search(r"Password recorded:\s*([A-Z]+)", output)
            if match:
                password = match.group(1)
                # Create a user-friendly representation
                password_display = password.replace('L', 'Left Eye ').replace('R', 'Right Eye ').replace('B', 'Both Eyes ')
                eyeblink_result = {'success': True, 'code': password, 'display': password_display}
            else:
                # Try JSON format
                match = re.search(r'{.*}', output)
                if match:
                    try:
                        json_str = match.group(0).replace("'", '"')
                        result_json = json.loads(json_str)
                        if 'password_string' in result_json:
                            password = result_json['password_string']
                            password_display = password.replace('L', 'Left Eye ').replace('R', 'Right Eye ').replace('B', 'Both Eyes ')
                            eyeblink_result = {'success': True, 'code': password, 'display': password_display}
                        else:
                            eyeblink_result = {'success': True, 'code': 'LB', 'display': 'Left Eye Both Eyes'}
                    except Exception as e:
                        print(f"JSON parsing error: {e}")
                        eyeblink_result = {'success': True, 'code': 'LB', 'display': 'Left Eye Both Eyes'}
                else:
                    eyeblink_result = {'success': True, 'code': 'LB', 'display': 'Left Eye Both Eyes'}
            
            return jsonify({
                'success': True,
                'complete': True,
                'code': eyeblink_result['code'],
                'display': eyeblink_result['display']
            })
            
        except Exception as e:
            print(f"Error running EBR.py: {str(e)}")
            eyeblink_result = {'success': False, 'error': str(e), 'code': 'LB', 'display': 'Left Eye Both Eyes'}
            return jsonify({
                'success': False,
                'error': str(e),
                'code': 'LB',
                'display': 'Left Eye Both Eyes'
            })
        finally:
            eyeblink_processing = False
    
    elif biometric_type == 'neck_movement':
        # Don't start if already processing
        if 'neck_movement_processing' in session and session['neck_movement_processing']:
            return jsonify({'success': True, 'in_progress': True})
    
        # Mark as processing
        session['neck_movement_processing'] = True

        try:
            # Kill any existing neck movement processes
            os.system("pkill -f 'python.*neckmov.py'")
        
            # Run neckmov.py with record flag and a single window flag
            result = subprocess.run(
                ['python', os.path.join(BASE_DIR, 'NeckMovement', 'neckmov.py'), '--record', '--single-window'],
                capture_output=True,
              text=True,
              timeout=120
          )
        
            # Process output
            output = result.stdout.strip()
            print(f"neckmov.py output: {output}")
        
            # Parse the password from output
            match = re.search(r"Password recorded:\s*([A-Z]+)", output)
            if match:
                neck_sequence = match.group(1)
                print(f"Neck movement sequence recorded: {neck_sequence}")
            
                # Create display version - e.g., "RLU" becomes "Right Left Up"
                directions = {
                    'U': 'Up',
                    'D': 'Down',
                    'L': 'Left',
                    'R': 'Right'
                }
                display_sequence = ' '.join([directions.get(char, char) for char in neck_sequence])
            
                # Success
                session['neck_movement_processing'] = False
                return jsonify({
                    'success': True,
                    'complete': True,
                    'code': neck_sequence,
                    'display': display_sequence
                })
            else:
                # Try JSON format
                match = re.search(r'{.*}', output)
                if match:
                    try:
                        json_str = match.group(0).replace("'", '"')
                        result_json = json.loads(json_str)
                        if 'password' in result_json:
                            neck_sequence = ''.join(result_json['password'])
                            directions = {
                                'U': 'Up',
                                'D': 'Down',
                                'L': 'Left',
                                'R': 'Right'
                            }
                            display_sequence = ' '.join([directions.get(char, char) for char in neck_sequence])
                        
                            session['neck_movement_processing'] = False
                            return jsonify({
                                'success': True,
                                'complete': True,
                                'code': neck_sequence,
                                'display': display_sequence
                            })
                    except Exception as e:
                        print(f"Error parsing JSON from neckmov.py output: {e}")
            
                # No pattern found in output, use default
                session['neck_movement_processing'] = False
                return jsonify({
                    'success': False,
                    'error': 'Failed to record neck movement pattern',
                    'code': 'RLU',  # Default pattern
                    'display': 'Right Left Up'  # Default pattern display
                })
        
        except Exception as e:
            print(f"Error running neckmov.py: {str(e)}")
            session['neck_movement_processing'] = False
            return jsonify({
                'success': False,
                'error': str(e),
                'code': 'RLU',  # Default pattern
                'display': 'Right Left Up'  # Default pattern display
            })
    elif biometric_type == 'voice':
        try:
            # Create a custom user folder for this user
            user_id = session['user_id']
            user_voice_dir = os.path.join(BASE_DIR, 'Voice', f'user_{user_id}')
            
            # Ensure the directory exists
            os.makedirs(user_voice_dir, exist_ok=True)
            
            # Create subdirectories
            os.makedirs(os.path.join(user_voice_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(user_voice_dir, 'test'), exist_ok=True)
            
            # Set environment variable - CRUCIAL for the voice modules
            print(f"Setting VOICE_FOLDER to: {user_voice_dir}")
            os.environ['VOICE_FOLDER'] = user_voice_dir
            
            # Return initial setup data
            voice_model_path = os.path.join(user_voice_dir, 'voice_model.pkl')
                    
            return jsonify({
                'success': True,
                'ready_to_record': True,
                'user_dir': user_voice_dir,
                'phrase': "I let the positive overrun the negative",
                'samples': 4,  # Changed from 6 to 4
                'code': voice_model_path
            })
                    
        except Exception as e:
            print(f"Error in voice recording setup: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
        


        
    elif biometric_type == 'face_detection':
        # Don't start if already processing
        if 'face_detection_processing' in session and session['face_detection_processing']:
            return jsonify({'success': False, 'error': 'Face detection already in progress'})

        # Mark as processing
        session['face_detection_processing'] = True

        try:
            # Get user ID for face enrollment
            user_id = session['user_id']
            user_name = f"user_{user_id}"  # Use user_id as identifier
            
            # Important: Set paths correctly to store faces within the FaceRecognition folder
            face_rec_dir = os.path.join(BASE_DIR, 'FaceRecognition')
            faces_dir = os.path.join(face_rec_dir, 'Faces', user_name)
            
            # Clean up existing face data for this user before re-enrollment
            if os.path.exists(faces_dir):
                print(f"Removing existing face data for user: {user_name}")
                shutil.rmtree(faces_dir)
            
            # Create user face directory
            os.makedirs(faces_dir, exist_ok=True)
            
            # Path for storing face model
            encodings_dir = os.path.join(face_rec_dir, 'encodings')
            os.makedirs(encodings_dir, exist_ok=True)
            face_model_path = os.path.join(encodings_dir, 'encodings.pkl')
            
            # Path to model weights file
            model_weights_path = os.path.join(face_rec_dir, 'facenet_keras_weights.h5')
            
            # First check if encodings.pkl exists, and if so, remove the user from it
            if os.path.exists(face_model_path):
                try:
                    # Load existing encodings
                    with open(face_model_path, 'rb') as f:
                        encoding_dict = pickle.load(f)
                    
                    # Remove this user if they exist in the encodings
                    if user_name in encoding_dict:
                        print(f"Removing existing encoding for user: {user_name}")
                        del encoding_dict[user_name]
                        
                        # Save the updated encodings
                        with open(face_model_path, 'wb') as f:
                            pickle.dump(encoding_dict, f)
                except Exception as e:
                    print(f"Error updating existing encodings file: {e}")
                    # If there's an error with the existing encodings file,
                    # we'll just remove it and let the training create a new one
                    if os.path.exists(face_model_path):
                        os.remove(face_model_path)
            
            # Use subprocess to run the faceenroll.py script in a new process
            print(f"Launching face enrollment for user: {user_name}")
            process = subprocess.Popen([
                sys.executable, 
                os.path.join(face_rec_dir, 'faceenroll.py'),
                '--name', user_name,
                '--output', os.path.join(face_rec_dir, 'Faces')  # Explicitly set output to the correct folder
            ])
            
            # Wait for the enrollment process to complete with timeout
            print("Waiting for face enrollment to complete...")
            process.wait(timeout=180)  # 3 minute timeout for enrollment
            
            # Now run the trainmodel.py script to train on all collected face data
            print("Face enrollment complete. Starting model training...")
            train_process = subprocess.Popen([
                sys.executable,
                os.path.join(face_rec_dir, 'trainmodel.py'),
                '--face-data', os.path.join(face_rec_dir, 'Faces'),
                '--output', face_model_path,
                '--model-weights', model_weights_path  # Explicitly pass the model weights path
            ])
            
            # Wait for training to complete with timeout
            print("Waiting for face model training to complete...")
            train_process.wait(timeout=300)  # 5 minute timeout for training
            
            print("Face model training complete.")
            
            # Clean up session flag
            session['face_detection_processing'] = False
            
            # Return success and the model path
            return jsonify({
                'success': True, 
                'code': face_model_path,
                'message': 'Face enrollment and training completed successfully'
            })
            
        except subprocess.TimeoutExpired as e:
            session['face_detection_processing'] = False
            print(f"Timeout during face detection process: {e}")
            return jsonify({
                'success': False, 
                'error': 'Face detection process timed out. Please try again.'
            })
            
        except Exception as e:
            session['face_detection_processing'] = False
            print(f"Error during face detection: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)})
# Add endpoint to check eyeblink recording status
@app.route('/eyeblink_status', methods=['GET'])
def eyeblink_status():
    global eyeblink_processing, eyeblink_result
    
    if not eyeblink_processing and eyeblink_result:
        # Recording is complete
        result = eyeblink_result
        return jsonify({
            'active': False,
            'complete': True,
            'success': result.get('success', False),
            'code': result.get('code', 'LB')
        })
    
    return jsonify({
        'active': eyeblink_processing,
        'complete': False
    })

# New endpoint: Browse File using Tkinter.
@app.route('/browse_file', methods=['GET'])
def browse_file():
    # Import Tkinter inside the function to avoid issues if not in GUI environment.
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        root = Tk()
        root.withdraw()  # Hide the main window.
        file_path = askopenfilename()
        root.destroy()
    except Exception as e:
        return jsonify({'file_path': '', 'error': str(e)})
    return jsonify({'file_path': file_path})

# Add this new route to handle temporary file uploads
@app.route('/upload_temp_file', methods=['POST'])
def upload_temp_file():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
        
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
        
    try:
        # Create a secure filename with a UUID to avoid collisions
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(TEMP_UPLOAD_FOLDER, filename)
        
        # Save the file
        file.save(file_path)
        
        # Return the server path to be used in the protection process
        return jsonify({'success': True, 'server_path': file_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Update the record_voice_sample endpoint
@app.route('/record_voice_sample', methods=['POST'])
def record_voice_sample():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    sample_number = data.get('sample_number', 1)
    user_dir = data.get('user_dir')
    microphone_id = data.get('microphone_id', 'default')
    
    # Validate user directory belongs to current user
    user_id = session['user_id']
    expected_dir = os.path.join(BASE_DIR, 'Voice', f'user_{user_id}')
    if user_dir != expected_dir:
        return jsonify({'success': False, 'error': 'Invalid user directory'})
    
    try:
        # Set environment variables
        os.environ['VOICE_FOLDER'] = user_dir
        if microphone_id != 'default':
            os.environ['MICROPHONE_ID'] = microphone_id
        
        # Import modules needed for recording
        import sounddevice as sd
        import soundfile as sf
        from datetime import datetime
        
        # Configuration
        PHRASE = "I let the positive overrun the negative"
        SAMPLE_RATE = 48000
        DURATION = 5
        
        # Create train directory if not exists
        train_dir = os.path.join(user_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"train_{sample_number}_{timestamp}.wav"
        output_file = os.path.join(train_dir, filename)
        
        # Configure recording device if specified
        device_kwargs = {}
        if microphone_id != 'default':
            device_kwargs['device'] = microphone_id
        
        # Record the audio
        print(f"Recording sample {sample_number} with microphone: {microphone_id}")
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            **device_kwargs
        )
        
        # Wait for recording to complete
        sd.wait()
        
        # Save the recording
        sf.write(output_file, recording, SAMPLE_RATE)
        
        # If this is the last sample, create augmented samples and train the model
        if sample_number == 4:
            # Import needed modules
            import numpy as np
            import pickle
            import random
            import librosa
            from scipy.signal import butter, filtfilt
            from Voice.voicerecord import extract_enhanced_features
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
            
            # Add noise reduction function
            def reduce_noise(y, sr):
                """Apply high-pass filter to reduce low frequency noise"""
                cutoff = 80  # Cut off frequency below 80Hz
                nyquist = 0.5 * sr
                normal_cutoff = cutoff / nyquist
                order = 4
                b, a = butter(order, normal_cutoff, btype='high', analog=False)
                y_filtered = filtfilt(b, a, y)
                return y_filtered
            
            # Add augmentation function
            def augment_audio(audio_file, output_file, augmentation_type='pitch'):
                """Create augmented versions of audio for better model training"""
                y, sr = librosa.load(audio_file, sr=None)
                
                # Apply noise reduction
                y = reduce_noise(y, sr)
                
                if augmentation_type == 'pitch':
                    # Slightly alter pitch
                    pitch_shift = random.uniform(-0.05, 0.05) 
                    y_augmented = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
                elif augmentation_type == 'speed':
                    # Slightly alter speed
                    speed_factor = random.uniform(0.995, 1.005)
                    y_augmented = librosa.effects.time_stretch(y, rate=speed_factor)
                else:
                    # Add small amount of noise
                    noise_level = 0.0002
                    noise = np.random.normal(0, noise_level, len(y))
                    y_augmented = y + noise
                
                # Make sure directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Save the file
                sf.write(output_file, y_augmented, sr)
                
                # Confirm file was saved
                file_size = os.path.getsize(output_file)
                print(f"✅ Augmented file saved: {os.path.basename(output_file)} ({file_size} bytes)")
                
                return output_file
            
            # Get all original training files
            original_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) 
                             if f.endswith('.wav') and 'aug_' not in f]
            
            # Create augmented versions of each file
            augmented_files = []
            print(f"Creating augmented samples from {len(original_files)} original recordings...")
            
            for i, orig_file in enumerate(original_files):
                # Create a new timestamp for each augmented file
                aug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                # Alternate between pitch and speed augmentation
                if i % 2 == 0:
                    aug_file = os.path.join(train_dir, f"train_aug_pitch_{i}_{aug_timestamp}.wav")
                    augmented_files.append(augment_audio(orig_file, aug_file, 'pitch'))
                else:
                    aug_file = os.path.join(train_dir, f"train_aug_speed_{i}_{aug_timestamp}.wav")
                    augmented_files.append(augment_audio(orig_file, aug_file, 'speed'))
            
            # Extract features from all files - both original and augmented
            features = []
            all_files = original_files + augmented_files
            
            print(f"Extracting features from {len(all_files)} files (original + augmented)...")
            for file in all_files:
                feature = extract_enhanced_features(file)
                features.append(feature)
                print(f"Extracted features from {os.path.basename(file)}")
            
            # Train the model
            features = np.array(features)
            print(f"Training model with {len(features)} samples, feature dimension: {features.shape}")
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create GMM with proper parameters for better model quality
            gmm = GaussianMixture(
                n_components=min(2, len(features)-1),
                covariance_type='diag',
                n_init=10,
                reg_covar=1.0,
                random_state=42,
                max_iter=300
            )
            
            # Fit the model
            gmm.fit(features_scaled)
            
            # Save the model
            model_path = os.path.join(user_dir, "voice_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump((gmm, scaler, PHRASE), f)
            
            print(f"Voice model trained and saved to {model_path}")
            
            return jsonify({
                'success': True,
                'sample_number': sample_number,
                'filename': filename,
                'model_saved': True,
                'model_path': model_path,
                'total_samples': len(all_files),
                'augmented_samples': len(augmented_files)
            })
        else:
            return jsonify({
                'success': True,
                'sample_number': sample_number,
                'filename': filename,
                'model_saved': False
            })
            
    except Exception as e:
        print(f"Error recording voice sample {sample_number}: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full stack trace
        return jsonify({
            'success': False,
            'error': str(e),
            'sample_number': sample_number
        })

# Ensure the voice_verify_sample endpoint in app.py is properly handling sample recording
@app.route('/voice_verify_sample', methods=['POST'])
def voice_verify_sample():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    sample_number = data.get('sample_number', 1)
    file_id = data.get('file_id')
    
    # Validate file access
    file_entry = get_file_by_id(file_id)
    if not file_entry or file_entry['user_id'] != session['user_id']:
        return jsonify({'success': False, 'error': 'File not found or access denied'})
    
    user = get_user_by_id(session['user_id'])
    voice_model_path = user['biometrics'].get('voice')
    
    if not voice_model_path or not os.path.exists(voice_model_path):
        return jsonify({'success': False, 'error': 'No voice biometric data found'})
    
    # Get the user directory from the model path
    user_voice_dir = os.path.dirname(voice_model_path)
    
    try:
        # Set environment variable
        os.environ['VOICE_FOLDER'] = user_voice_dir
        
        # Import modules needed for recording
        import sounddevice as sd
        import soundfile as sf
        from datetime import datetime
        
        # Create test directory if not exists
        test_dir = os.path.join(user_voice_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Configuration from voicevalidate.py
        SAMPLE_RATE = 48000
        DURATION = 5
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_{sample_number}_{timestamp}.wav"
        output_file = os.path.join(test_dir, filename)
        
        # Record the audio
        print(f"Recording verification sample {sample_number}...")
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1
        )
        
        # Wait for recording to complete
        sd.wait()
        
        # Save the recording
        sf.write(output_file, recording, SAMPLE_RATE)
        
        return jsonify({
            'success': True,
            'sample_number': sample_number,
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error recording voice verification sample {sample_number}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'sample_number': sample_number
        })

# Index route: redirect to login.
@app.route("/")
def index():
    return redirect(url_for('login'))

# -----------------------
# User Registration & Login
# -----------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password_hash = generate_password_hash(request.form['password'])
        user_pin = request.form.get('pin', '')
        # Make PIN compulsory.
        if not user_pin:
            flash("Account PIN is required.", "danger")
            return redirect(url_for('register'))
        if get_user_by_username(username):
            flash("Username already exists", "danger")
            return redirect(url_for('register'))
        data = read_data()
        user = {
            "id": str(uuid.uuid4()),
            "username": username,
            "password": password_hash,
            "pin": user_pin,
            "biometrics": {}
        }
        data['users'].append(user)
        write_data(data)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user_by_username(username)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            flash("Logged in successfully!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')

# -----------------------
# Add Biometrics
# -----------------------
@app.route('/add_biometrics', methods=['GET', 'POST'])
def add_biometrics():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    
    user = get_user_by_id(session['user_id'])
    
    if request.method == 'POST':
        # Process all biometrics at once
        biometrics = {}
        
        # Process fingerprint
        if request.form.get('fingerprint'):
            biometrics['fingerprint'] = request.form.get('fingerprint')
        else:
            # If no fingerprint provided in the form, check if there's a file upload
            if 'fingerprint_file' in request.files:
                fingerprint_file = request.files['fingerprint_file']
                if fingerprint_file.filename:
                    # Import the enroll module
                    from FingerPrint.enroll import save_fingerprint
                    
                    # Save the fingerprint image
                    try:
                        user_id = session['user_id']
                        fingerprint_path = save_fingerprint(fingerprint_file, user_id)
                        biometrics['fingerprint'] = fingerprint_path
                    except Exception as e:
                        print(f"Error saving fingerprint: {str(e)}")
                        biometrics['fingerprint'] = fingerprintsave.save(None)  # Fallback
                else:
                    biometrics['fingerprint'] = fingerprintsave.save(None)
            else:
                biometrics['fingerprint'] = fingerprintsave.save(None)
            
        if request.form.get('eyeblink'):
            biometrics['eyeblink'] = request.form.get('eyeblink')
        else:
            # Use EBR.py directly for eyeblink
            try:
                # Kill any existing EBR processes first
                os.system("pkill -f 'python.*EBR.py'")
                
                # Run EBR.py with --record flag
                result = subprocess.run(
                    ['python', os.path.join(BASE_DIR, 'EyeBlink', 'EBR.py'), '--record', '--headless'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Parse the output
                output = result.stdout.strip()
                print(f"EBR.py output: {output}")
                
                # Extract the password
                match = re.search(r"Password recorded:\s*([A-Z]+)", output)
                if match:
                    biometrics['eyeblink'] = match.group(1)
                else:
                    # Try JSON format
                    match = re.search(r'{.*}', output)
                    if match:
                        try:
                            json_str = match.group(0).replace("'", '"')
                            result_json = json.loads(json_str)
                            if 'password_string' in result_json:
                                biometrics['eyeblink'] = result_json['password_string']
                            else:
                                biometrics['eyeblink'] = 'LB'  # Default fallback
                        except:
                            biometrics['eyeblink'] = 'LB'  # Default fallback
                    else:
                        biometrics['eyeblink'] = 'LB'  # Default fallback
            except Exception as e:
                print(f"Error running EBR.py: {str(e)}")
                biometrics['eyeblink'] = 'LB'  # Default fallback
            
        if request.form.get('neck_movement'):
            biometrics['neck_movement'] = request.form.get('neck_movement')
        else:
            biometrics['neck_movement'] = neckmovementsave.save(None)
            
        if request.form.get('voice'):
            biometrics['voice'] = request.form.get('voice')
        else:
            biometrics['voice'] = voicesave.save(None)
            
        if request.form.get('face_detection'):
            biometrics['face_detection'] = request.form.get('face_detection')
        else:
            # Create user-specific face encoding path
            user_id = session['user_id']
            user_name = f"user_{user_id}"
            face_model_path = os.path.join(BASE_DIR, 'encodings', f"{user_name}_encoding.pkl")
            
            # Check if the model file exists
            if os.path.exists(face_model_path):
                biometrics['face_detection'] = face_model_path
            else:
                # If not, use the simple placeholder from facedetectionsave
                biometrics['face_detection'] = facedetectionsave.save(None)
        
        # Save all biometrics at once
        user['biometrics'] = biometrics
        update_user(user)
        
        flash("Biometrics added successfully.", "success")
        return redirect(url_for('dashboard'))
    
    return render_template('add_biometrics.html')

# -----------------------
# Dashboard & File Protection
# -----------------------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    data = read_data()
    user_files = [f for f in data['files'] if f['user_id'] == session['user_id']]
    user = get_user_by_id(session['user_id'])
    return render_template('dashboard.html', files=user_files, user=user)




@app.route('/protect_file', methods=['GET', 'POST'])
def protect_file():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
        
    user = get_user_by_id(session['user_id'])
    if not user.get('biometrics'):
        flash("Please add your biometrics first.", "danger")
        return redirect(url_for('add_biometrics'))
        
    if request.method == 'POST':
        user_id = session['user_id']
        
        # Check if we have file_path from a previous upload
        file_path = request.form.get('file_path')
        
        # If no file_path, check if a file was uploaded directly
        if not file_path and 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                # Create a temporary file
                temp_filename = f"{uuid.uuid4()}_{secure_filename(uploaded_file.filename)}"
                file_path = os.path.join(TEMP_UPLOAD_FOLDER, temp_filename)
                uploaded_file.save(file_path)
        
        if not file_path or not os.path.exists(file_path):
            flash("No file selected or file not found.", "danger")
            return redirect(url_for('protect_file'))
        
        # Store original filename without UUID prefixes for display
        original_filename = os.path.basename(file_path)
        # If it's from temp uploads, remove the UUID prefix for display
        if TEMP_UPLOAD_FOLDER in file_path:
            # Extract the original filename without the UUID prefix
            name_parts = original_filename.split('_', 1)
            if len(name_parts) > 1:
                original_filename = name_parts[1]
        
        # Create a unique encrypted file path
        encrypted_filename = f"{user_id}_{uuid.uuid4()}_{original_filename}.encrypted"
        destination_path = os.path.join(PROTECTED_FOLDER, encrypted_filename)
        
        try:
            # Encrypt the file
            encrypt_success = encrypt_file(file_path, destination_path)
            
            if not encrypt_success:
                flash("Failed to encrypt file.", "danger")
                return redirect(url_for('protect_file'))
            
            # Remove the original file if it's not in our temp uploads folder
            # Always attempt to remove the original file now
            if TEMP_UPLOAD_FOLDER not in file_path:
                try:
                    # Confirm the file exists before trying to remove it
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Original file removed: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove original file: {e}")
            else:
                # If it's a temp file, still remove it after encryption
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file: {e}")
                    
            flash("File encrypted and secured successfully.", "success")
            
        except Exception as e:
            flash(f"Error protecting file: {str(e)}", "danger")
            return redirect(url_for('protect_file'))
            
        # Create file entry record - use original_filename instead of path basename
        file_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "file_name": original_filename,  # Use cleaned up original name
            "file_path": destination_path,
            "encrypted": True,
            "require_pin": True if request.form.get('require_pin') else False,
            "require_fingerprint": True if request.form.get('require_fingerprint') else False,
            "require_eyeblink": True if request.form.get('require_eyeblink') else False,
            "require_neck_movement": True if request.form.get('require_neck_movement') else False,
            "require_voice": True if request.form.get('require_voice') else False,
            "require_face_detection": True if request.form.get('require_face_detection') else False
        }
        
        data = read_data()
        data['files'].append(file_entry)
        write_data(data)
        
        flash("File protected successfully", "success")
        return redirect(url_for('dashboard'))
        
    return render_template('protect_file.html')

# -----------------------
# File Access & Multi-Step Verification
# -----------------------
@app.route('/verify_auth/<file_id>', methods=['POST'])
def verify_auth(file_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    file_entry = get_file_by_id(file_id)
    if not file_entry or file_entry['user_id'] != session['user_id']:
        return jsonify({'success': False, 'error': 'File not found or access denied'})
    
    user = get_user_by_id(session['user_id'])
    methods = ['fingerprint', 'eyeblink', 'neck_movement', 'voice', 'face_detection']
    for method in methods:
        if file_entry.get(f"require_{method}"):
            if method == "fingerprint":
                result = fingerprint.verify(None, user['biometrics'].get('fingerprint'))
                if result.strip() != "Access granted":
                    return jsonify({'success': False, 'message': 'Fingerprint verification failed'})
            elif method == "eyeblink":
                # Use EBR.py for eyeblink verification
                try:
                    stored_pattern = user['biometrics'].get('eyeblink')
                    if not stored_pattern:
                        return jsonify({'success': False, 'error': 'No eye blink pattern stored'})
                    
                    # Kill any existing EBR processes
                    os.system("pkill -f 'python.*EBR.py'")
                    
                    # Run EBR.py with --verify flag
                    result = subprocess.run(
                        ['python', os.path.join(BASE_DIR, 'EyeBlink', 'EBR.py'), 
                         '--verify', stored_pattern, '--timeout', '60'],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    # Process output
                    output = result.stdout.strip()
                    print(f"EBR.py verification output: {output}")
                    
                    verification_successful = False
                    if "verified': True" in output or "Password verification successful" in output:
                        verification_successful = True
                    else:
                        # Try JSON format
                        match = re.search(r'{.*}', output)
                        if match:
                            try:
                                json_str = match.group(0).replace("'", '"')
                                result_json = json.loads(json_str)
                                if result_json.get('verified', False):
                                    verification_successful = True
                            except Exception as e:
                                print(f"JSON parsing error: {e}")
                    
                    if not verification_successful:
                        return jsonify({'success': False, 'message': 'Eyeblink verification failed'})
                except Exception as e:
                    print(f"Error in EBR verification: {str(e)}")
                    return jsonify({'success': False, 'message': f'Eyeblink verification error: {str(e)}'})
            elif method == "neck_movement":
                try:
                    stored_pattern = user['biometrics'].get('neck_movement')
                    if not stored_pattern:
                        return jsonify({'success': False, 'error': 'No neck movement pattern stored'})
                    
                    # Kill any existing neckmov processes
                    os.system("pkill -f 'python.*neckmov.py'")
                    
                    # Run neckmov.py with --verify flag
                    result = subprocess.run(
                        ['python', os.path.join(BASE_DIR, 'NeckMovement', 'neckmov.py'), 
                         '--verify', stored_pattern, '--timeout', '60'],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    # Process output
                    output = result.stdout.strip()
                    print(f"neckmov.py verification output: {output}")
                    
                    verification_successful = False
                    if "verified': True" in output or "Verification successful" in output:
                        verification_successful = True
                    else:
                        # Try JSON format
                        match = re.search(r'{.*}', output)
                        if match:
                            try:
                                json_str = match.group(0).replace("'", '"')
                                result_json = json.loads(json_str)
                                if result_json.get('verified', False):
                                    verification_successful = True
                            except Exception as e:
                                print(f"JSON parsing error: {e}")
                    
                    if not verification_successful:
                        return jsonify({'success': False, 'message': 'Neck movement verification failed'})
                except Exception as e:
                    print(f"Error in neck movement verification: {str(e)}")
                    return jsonify({'success': False, 'message': f'Neck movement verification error: {str(e)}'})
            elif method == "voice":
                try:
                    # Get the stored voice model path from user biometrics
                    voice_model_path = user['biometrics'].get('voice')
                    if not voice_model_path or not os.path.exists(voice_model_path):
                        return jsonify({'success': False, 'error': 'No voice biometric data found'})
                    
                    # Get the user directory from the model path
                    user_voice_dir = os.path.dirname(voice_model_path)
                    
                    # Set environment variable - this is critical for the voice modules
                    print(f"Setting VOICE_FOLDER to: {user_voice_dir}")
                    os.environ['VOICE_FOLDER'] = user_voice_dir
                    
                    try:
                        # Import and use the verify_voice_biometrics function directly
                        from Voice.voicevalidate import verify_voice_biometrics
                        
                        # Capture output
                        original_stdout = sys.stdout
                        output_capture = io.StringIO()
                        sys.stdout = output_capture
                        
                        # Verify the voice - get the result and score info
                        verification_result = verify_voice_biometrics()
                        
                        # Convert numpy bool_ to Python bool if needed
                        if hasattr(verification_result, 'item'):
                            verification_successful = bool(verification_result.item())
                        else:
                            verification_successful = bool(verification_result)
                        
                        # Get captured output
                        output = output_capture.getvalue()
                        
                        # Extract scores from the output
                        import re
                        avg_score_match = re.search(r'Average score: ([-\d.]+)', output)
                        threshold_match = re.search(r'Threshold: ([-\d.]+)', output)
                        
                        avg_score = float(avg_score_match.group(1)) if avg_score_match else None
                        threshold = float(threshold_match.group(1)) if threshold_match else None
                        
                        # Restore stdout
                        sys.stdout = original_stdout
                        
                        if not verification_successful:
                            return jsonify({'success': False, 'message': 'Voice verification failed'})
                    except Exception as e:
                        # Make sure to restore stdout
                        if 'original_stdout' in locals():
                            sys.stdout = original_stdout
                        print(f"Error in voice verification function: {str(e)}")
                        return jsonify({'success': False, 'message': f'Voice verification error: {str(e)}'})
                except Exception as e:
                    print(f"Error in voice verification: {str(e)}")
                    return jsonify({'success': False, 'message': f'Voice verification error: {str(e)}'})
            elif method == "face_detection":
                result = facedetection.verify(None, user['biometrics'].get('face_detection'))
                if result.strip() != "Access granted":
                    return jsonify({'success': False, 'message': 'Face detection verification failed'})
    
    return jsonify({'success': True})

@app.route('/open_file/<file_id>', methods=['GET', 'POST'])
def open_file_route(file_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    file_entry = get_file_by_id(file_id)
    if not file_entry or file_entry['user_id'] != session['user_id']:
        flash("File not found or access denied", "danger")
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Check if PIN is required and provided
        user = get_user_by_id(session['user_id'])
        user_pin = user.get('pin', '')  # This is decrypted by get_user_by_id
        
        # If PIN is required, verify it
        if file_entry.get('require_pin', False):
            pin = request.form.get('pin', '')
            if pin != user_pin:
                flash("Incorrect PIN", "danger")
                return redirect(url_for('open_file_route', file_id=file_id))
        
        # Check if we should download or view directly
        mode = request.form.get('access_mode', 'view')
        
        # All checks passed - decrypt and return the file
        try:
            encrypted_file_path = file_entry['file_path']
            if not os.path.exists(encrypted_file_path):
                flash("Encrypted file not found on server", "danger")
                return redirect(url_for('dashboard'))
            
            # Get original file extension
            file_name = file_entry['file_name']
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # Create a temporary directory for decrypted files if it doesn't exist
            temp_decrypted_dir = os.path.join(BASE_DIR, "temp_decrypted")
            os.makedirs(temp_decrypted_dir, exist_ok=True)
            
            # Create a unique temporary file path for the decrypted file
            temp_decrypted_path = os.path.join(temp_decrypted_dir, f"decrypted_{uuid.uuid4()}{file_ext}")
            
            # Decrypt the file to the temporary path
            if not decrypt_file(encrypted_file_path, temp_decrypted_path):
                flash("Failed to decrypt file", "danger")
                return redirect(url_for('dashboard'))
            
            # Schedule file deletion after 2 minutes (120 seconds)
            def delete_temp_file(filepath, delay=120):
                time.sleep(delay)
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"Temporary decrypted file removed: {filepath}")
                except Exception as e:
                    print(f"Error removing temp file: {str(e)}")
            
            # Start background thread to delete the file
            threading.Thread(target=delete_temp_file, args=(temp_decrypted_path,), daemon=True).start()
            
            # Return the file based on mode
            if mode == 'download':
                return send_file(
                    temp_decrypted_path,
                    as_attachment=True,
                    download_name=file_name
                )
            else:
                # View mode - determine appropriate content type
                mime_type = None
                if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    mime_type = f'image/{file_ext[1:]}'
                elif file_ext == '.pdf':
                    mime_type = 'application/pdf'
                
                if mime_type:
                    return send_file(temp_decrypted_path, mimetype=mime_type)
                else:
                    # For other file types, default to download
                    return send_file(
                        temp_decrypted_path,
                        as_attachment=True,
                        download_name=file_name
                    )
            
        except Exception as e:
            flash(f"Error accessing file: {str(e)}", "danger")
            return redirect(url_for('dashboard'))
    
    # GET request - display the verification UI
    return render_template("open_file.html", file_entry=file_entry)

# -----------------------
# Reset Biometrics Routes
# -----------------------
@app.route('/reset_biometrics')
def reset_biometrics():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    user = get_user_by_id(session['user_id'])
    available = []
    for bio in ['fingerprint', 'eyeblink', 'neck_movement', 'voice', 'face_detection']:
        if user['biometrics'].get(bio):
            available.append(bio)
    return render_template('reset_biometrics.html', biometrics=available)

@app.route('/reset_biometrics/<bio_type>', methods=['GET', 'POST'])
def reset_bio(bio_type):
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    if bio_type not in ['fingerprint', 'eyeblink', 'neck_movement', 'voice', 'face_detection']:
        flash("Invalid biometric type", "danger")
        return redirect(url_for('reset_biometrics'))
    if request.method == 'POST':
        # Check if the biometric data was submitted in the form
        if bio_type in request.form and request.form.get(bio_type):
            new_code = request.form.get(bio_type)
            print(f"Using submitted {bio_type} code: {new_code}")
        else:
            # Fallback to generating a new code if not provided in the form
            if bio_type == 'fingerprint':
                new_code = fingerprintsave.save(None)
            elif bio_type == 'eyeblink':
                # Use EBR.py for eyeblink reset
                try:
                    # Run the EBR.py script with --record flag and capture output
                    result = subprocess.run(
                        ['python', os.path.join(BASE_DIR, 'EyeBlink', 'EBR.py'), '--record', '--headless'],
                        capture_output=True,
                        text=True,
                        timeout=60  # Add timeout to prevent hanging
                    )
                    
                    # Parse the output
                    output = result.stdout.strip()
                    print(f"EBR.py output: {output}")  # For debugging
                    
                    # Look for password_string in different formats
                    # First try direct regex pattern
                    match = re.search(r"Password recorded:\s*([A-Z]+)", output)
                    if match:
                        new_code = match.group(1)
                    # Then try JSON extraction
                    else:
                        match = re.search(r'{.*}', output)
                        if match:
                            try:
                                json_str = match.group(0).replace("'", '"')
                                result_json = json.loads(json_str)
                                if 'password_string' in result_json:
                                    new_code = result_json['password_string']
                                else:
                                    new_code = ''.join(result_json.get('password', ['L', 'B']))
                            except:
                                # Default fallback
                                new_code = 'LBRL'
                        else:
                            # Default fallback
                            new_code = 'LBRL'
                except Exception as e:
                    print(f"Error running EBR.py for reset: {str(e)}")
                    flash(f"Error resetting eyeblink: {str(e)}", "danger")
                    return redirect(url_for('reset_biometrics'))
            elif bio_type == 'neck_movement':
                new_code = neckmovementsave.save(None)
            elif bio_type == 'voice':
                new_code = voicesave.save(None)
            elif bio_type == 'face_detection':
                if request.form.get('biometricData'):
                    new_code = request.form.get('biometricData')
                    
                    # Check if this is a special placeholder value indicating failure
                    if new_code == 'default_face':
                        # Keep the existing face data if enrollment failed
                        user = get_user_by_id(session['user_id'])
                        new_code = user['biometrics'].get('face_detection', 'default_face')
                else:
                    # The reset happened successfully through the record_biometric call
                    # and the path to the new face model should be in the form data
                    new_code = request.form.get('biometricData', 'default_face')
        
        # Update user biometrics with the new code
        user = get_user_by_id(session['user_id'])
        user['biometrics'][bio_type] = new_code
        update_user(user)
        flash(f"{bio_type.replace('_', ' ').capitalize()} biometric reset successfully.", "success")
        return redirect(url_for('dashboard'))
    return render_template('reset_bio.html', bio_type=bio_type)

# Add endpoint to stop camera explicitly
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'success': True})

@app.route('/verify_biometric', methods=['POST'])
def verify_biometric():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    # Get the biometric type from form or JSON data
    biometric_type = request.form.get('biometric_type')
    file_id = request.form.get('file_id')
    
    # If not in form data, try to get from JSON
    if (not biometric_type or not file_id) and request.is_json:
        data = request.get_json()
        biometric_type = data.get('biometric_type')
        file_id = data.get('file_id')
    
    file_entry = get_file_by_id(file_id)
    if not file_entry or file_entry['user_id'] != session['user_id']:
        return jsonify({'success': False, 'error': 'File not found or access denied'})
    
    user = get_user_by_id(session['user_id'])
    
    if biometric_type == 'fingerprint':
        # Access the stored fingerprint data
        stored_fingerprint = user['biometrics'].get('fingerprint')
        if not stored_fingerprint:
            return jsonify({'success': False, 'error': 'No fingerprint data found'})
            
        # Check if a fingerprint image file is included
        if 'fingerprint_image' in request.files:
            fingerprint_file = request.files['fingerprint_image']
            
            # Get sequence information
            sequence_number = int(request.form.get('sequence_number', 1))
            
            # Import the verify module
            from FingerPrint.everify import verify_fingerprint
            
            # Verify the fingerprint
            try:
                user_id = session['user_id']
                result = verify_fingerprint(fingerprint_file, user_id, sequence_number)
                
                # Ensure we handle the returned data properly, with defaults for missing values
                total_prints = result.get('total_fingerprints', 1)
                current_seq = result.get('current_sequence', sequence_number)
                is_last = result.get('is_last', current_seq >= total_prints)
                
                return jsonify({
                    'success': bool(result.get('success', False)),
                    'similarity': float(result.get('similarity_percent', 0.0)),
                    'threshold': float(result.get('threshold_percent', 0.0)),
                    'message': str(result.get('message', '')),
                    'total_fingerprints': total_prints,
                    'current_sequence': current_seq,
                    'is_last': is_last
                })
            except Exception as e:
                print(f"Error verifying fingerprint: {str(e)}")
                traceback.print_exc()
                return jsonify({
                    'success': False, 
                    'error': f"Fingerprint verification failed: {str(e)}"
                })
        
        # If no file is uploaded, return error
        return jsonify({'success': False, 'error': 'No fingerprint image provided'})
    elif biometric_type == 'eyeblink':
        try:
            # Get the stored eye blink pattern
            stored_pattern = user['biometrics'].get('eyeblink')
            if not stored_pattern:
                return jsonify({'success': False, 'error': 'No eye blink pattern stored'})
            
            # Kill any existing EBR processes to avoid duplicate windows
            os.system("pkill -f 'python.*EBR.py'")
            
            # Use subprocess to run EBR.py with --verify flag and the stored pattern
            result = subprocess.run(
                ['python', os.path.join(BASE_DIR, 'EyeBlink', 'EBR.py'), 
                 '--verify', stored_pattern, '--timeout', '60'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Process output
            output = result.stdout.strip()
            print(f"EBR.py verification output: {output}")
            
            # Parse verification result from output
            if "verified': True" in output or "Password verification successful" in output:
                return jsonify({'success': True})
            # Try JSON format
            match = re.search(r'{.*}', output)
            if match:
                try:
                    json_str = match.group(0).replace("'", '"')
                    result_json = json.loads(json_str)
                    if result_json.get('verified', False):
                        return jsonify({'success': True})
                except Exception as e:
                    print(f"JSON parsing error: {e}")
            
            return jsonify({'success': False})
                
        except Exception as e:
            print(f"Error in EBR verification: {str(e)}")
            return jsonify({'success': False, 'error': str(e)})
    elif biometric_type == 'neck_movement':
        try:
            stored_pattern = user['biometrics'].get('neck_movement')
            if not stored_pattern:
                return jsonify({'success': False, 'error': 'No neck movement pattern stored'})
        
        # Kill any existing neckmov processes
            os.system("pkill -f 'python.*neckmov.py'")
        
        # Run neckmov.py with --verify flag and single-window flag
            result = subprocess.run(
                ['python', os.path.join(BASE_DIR, 'NeckMovement', 'neckmov.py'), 
                '--verify', stored_pattern, '--timeout', '60', '--single-window'],
                capture_output=True,
                text=True,
                timeout=120
            )
        
        # Process output
            output = result.stdout.strip()
            print(f"neckmov.py verification output: {output}")
        
            verification_successful = False
            if "verified': True" in output or "Verification successful" in output:
                verification_successful = True
            else:
            # Try JSON format
                match = re.search(r'{.*}', output)
                if match:
                    try:
                        json_str = match.group(0).replace("'", '"')
                        result_json = json.loads(json_str)
                        if result_json.get('verified', False):
                            verification_successful = True
                    except Exception as e:
                        print(f"JSON parsing error: {e}")
        
            if verification_successful:
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'message': 'Neck movement verification failed'})
        except Exception as e:
            print(f"Error in neck movement verification: {str(e)}")
            return jsonify({'success': False, 'message': f'Neck movement verification error: {str(e)}'})
    elif biometric_type == 'voice':
        # Similar logic for voice verification
        voice_model_path = user['biometrics'].get('voice')
        if not voice_model_path or not os.path.exists(voice_model_path):
            return jsonify({'success': False, 'error': 'No voice biometric data found'})
        
        # Get the user directory from the model path
        user_voice_dir = os.path.dirname(voice_model_path)
        
        # Set environment variable for the voice modules
        os.environ['VOICE_FOLDER'] = user_voice_dir
        
        try:
            # Import the verification function directly
            from Voice.voicevalidate import verify_voice_biometrics
            
            # Capture output with StringIO
            original_stdout = sys.stdout
            output_capture = io.StringIO()
            sys.stdout = output_capture
            
            # Run the verification and get the boolean result
            verification_result = verify_voice_biometrics()
            
            # Convert numpy bool_ to regular bool if needed
            if hasattr(verification_result, 'item'):
                verification_success = bool(verification_result.item())
            else:
                verification_success = bool(verification_result)
            
            # Get and parse the captured output
            output = output_capture.getvalue()
            sys.stdout = original_stdout  # Restore stdout
            
            return jsonify({
                'success': verification_success,
                'output': output,
                'output_lines': output.strip().split('\n') if output else []
            })
            
        except Exception as e:
            # Make sure stdout is restored
            if 'original_stdout' in locals():
                sys.stdout = original_stdout
            print(f"Error in voice verification function: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
    elif biometric_type == 'face_detection':
        # Get user ID for authentication
        user_id = session['user_id']
        user_name = f"user_{user_id}"
        
        # Get the stored biometric data path 
        face_model_path = user['biometrics'].get('face_detection')
        if not face_model_path or not os.path.exists(face_model_path):
            return jsonify({'success': False, 'error': 'No face biometric data found'})
        
        try:
            # Important: Set paths correctly
            face_rec_dir = os.path.join(BASE_DIR, 'FaceRecognition')
            # Use absolute path for model weights
            model_weights_path = os.path.join(face_rec_dir, 'facenet_keras_weights.h5')
            
            if not os.path.exists(model_weights_path):
                return jsonify({'success': False, 'error': f'Model weights file not found at: {model_weights_path}'})
            
            # Important: Kill any existing face detection processes first
            os.system("pkill -f 'python.*detectface.py'")
            
            # Create a temporary file to store results
            result_file = os.path.join(TEMP_UPLOAD_FOLDER, f"face_auth_result_{user_id}_{uuid.uuid4()}.txt")
            
            print(f"Starting face verification for user: {user_name}")
            print(f"Model path: {face_model_path}")
            print(f"Weights path: {model_weights_path}")
            
            # Change directory to the FaceRecognition folder to ensure relative paths work
            current_dir = os.getcwd()
            os.chdir(face_rec_dir)
            
            # Run detectface.py with its working directory set to FaceRecognition
            process = subprocess.Popen([
                sys.executable,
                'detectface.py',
                '--authenticate',
                '--person', user_name,
                '--encodings', face_model_path,
                '--model', model_weights_path,
                '--result-file', result_file
            ])
            
            # Restore original directory
            os.chdir(current_dir)
            
            # Start polling for the verification result
            return jsonify({
                'success': True,
                'message': 'Face verification process started',
                'result_file': result_file
            })
                
        except Exception as e:
            print(f"Error during face verification: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Error launching face verification: {str(e)}'})
    else:
        return jsonify({'success': False, 'error': 'Invalid biometric type'})

@app.route('/face_detection_status', methods=['GET'])
def face_detection_status():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    user_id = session['user_id']
    user_name = f"user_{user_id}"
    face_model_path = os.path.join(BASE_DIR, 'encodings', f"{user_name}_encoding.pkl")
    
    # Check if face enrollment is in progress
    processing = session.get('face_detection_processing', False)
    
    # Check if the model file has been created
    model_exists = os.path.exists(face_model_path)
    
    return jsonify({
        'active': processing,
        'complete': model_exists,
        'success': model_exists,
        'code': face_model_path if model_exists else None
    })

# Add this new endpoint after the verify_biometric endpoint

@app.route('/check_face_verification', methods=['POST'])
def check_face_verification():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    result_file = data.get('result_file')
    
    # Check if the file exists
    if not os.path.exists(result_file):
        # Check if the process is still running
        process_running = subprocess.run(
            f"pgrep -f 'python.*detectface.py'", 
            shell=True, 
            stdout=subprocess.PIPE
        ).stdout.decode().strip() != ''
        
        if process_running:
            return jsonify({'completed': False, 'message': 'Verification in progress'})
        else:
            # Process ended but no result file - something went wrong
            return jsonify({
                'completed': True, 
                'success': False, 
                'error': 'Face verification failed - no result data'
            })
    
    # Read the content of the result file
    try:
        with open(result_file, 'r') as f:
            content = f.read().strip()
        
        # Always clean up the file after reading
        try:
            os.remove(result_file)
        except:
            pass
        
        # Check for errors in the content
        if "Error" in content or "error" in content:
            return jsonify({
                'completed': True, 
                'success': False, 
                'error': content.split('Error:', 1)[1].strip() if 'Error:' in content else content
            })
        
        # Check if verification was successful
        if "Authentication successful" in content or "authenticated: True" in content:
            return jsonify({
                'completed': True, 
                'success': True, 
                'message': 'Face verification successful'
            })
        else:
            return jsonify({
                'completed': True, 
                'success': False, 
                'error': 'Face verification failed'
            })
            
    except Exception as e:
        print(f"Error reading verification result: {e}")
        return jsonify({
            'completed': True, 
            'success': False, 
            'error': f'Error reading result: {str(e)}'
        })

@app.route('/kill_face_detection', methods=['POST'])
def kill_face_detection():
    """Kill any running face detection processes"""
    try:
        os.system("pkill -f 'python.*detectface.py'")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
# Add this new route to app.py
@app.route('/privacy_policy')
def privacy_policy():
    return render_template('privacy_policy.html')


# Add a new route for PIN verification
@app.route('/verify_pin', methods=['POST'])
def verify_pin():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    pin = data.get('pin')
    file_id = data.get('file_id')
    
    # Verify the file exists and belongs to the user
    file_entry = get_file_by_id(file_id)
    if not file_entry or file_entry['user_id'] != session['user_id']:
        return jsonify({'success': False, 'message': 'File not found or access denied'})
    
    # Get the user's PIN
    user = get_user_by_id(session['user_id'])
    user_pin = user.get('pin', '')
    
    # Verify the PIN
    if pin == user_pin:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Incorrect PIN. Please try again.'})

# Hospital Dashboard setup route
@app.route('/hospital_dashboard_setup', methods=['GET', 'POST'])
def hospital_dashboard_setup():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    
    user = get_user_by_id(session['user_id'])
    
    if not user.get('biometrics'):
        flash("Please add your biometrics first.", "warning")
        return redirect(url_for('add_biometrics'))
    
    if request.method == 'POST':
        # Get the security settings from the form
        hospital_settings = {
            "require_pin": True if request.form.get('require_pin') else False,
            "require_fingerprint": True if request.form.get('require_fingerprint') else False,
            "require_eyeblink": True if request.form.get('require_eyeblink') else False,
            "require_neck_movement": True if request.form.get('require_neck_movement') else False,
            "require_voice": True if request.form.get('require_voice') else False,
            "require_face_detection": True if request.form.get('require_face_detection') else False,
        }
        
        # Check if at least one method is selected
        if not any(hospital_settings.values()):
            flash("Please select at least one security method.", "warning")
            return redirect(url_for('hospital_dashboard_setup'))
        
        # Update user settings
        is_update = 'hospital_settings' in user
        if not is_update:
            user['hospital_settings'] = {}
            
        user['hospital_settings'] = hospital_settings
        update_user(user)
        
        if is_update:
            flash("Hospital dashboard security settings updated successfully", "success")
        else:
            flash("Hospital dashboard security settings saved successfully", "success")
            
        return redirect(url_for('dashboard'))
        
    return render_template('hospital_dashboard_setup.html', user=user)

# Add route to access the hospital dashboard
@app.route('/access_hospital_dashboard', methods=['GET', 'POST'])
def access_hospital_dashboard():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    
    user = get_user_by_id(session['user_id'])
    
    # Check if hospital dashboard settings exist
    if not user.get('hospital_settings'):
        flash("Please configure hospital dashboard security settings first", "warning")
        return redirect(url_for('hospital_dashboard_setup'))
    
    # If user is already verified for hospital dashboard
    if session.get('hospital_access_granted'):
        return redirect(url_for('hospital_dashboard'))
    
    return render_template('access_hospital_dashboard.html', security_settings=user.get('hospital_settings', {}))

# Add route to verify hospital PIN
@app.route('/verify_hospital_pin', methods=['POST'])
def verify_hospital_pin():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    pin = data.get('pin')
    
    # Get the user's PIN
    user = get_user_by_id(session['user_id'])
    user_pin = user.get('pin', '')
    
    # Verify the PIN
    if pin == user_pin:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Incorrect PIN. Please try again.'})

# Add route to verify hospital biometrics
@app.route('/verify_hospital_biometric', methods=['POST'])
def verify_hospital_biometric():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    # Get the biometric type
    biometric_type = None
    
    # Try to get from form data first (for file uploads)
    if 'biometric_type' in request.form:
        biometric_type = request.form.get('biometric_type')
    # If not in form, try to get from JSON body
    elif request.is_json:
        data = request.get_json()
        biometric_type = data.get('biometric_type')
    
    # Get the user
    user = get_user_by_id(session['user_id'])
    
    # Based on biometric type, verify it
    if biometric_type == 'fingerprint':
        # Access the stored fingerprint data
        stored_fingerprint = user['biometrics'].get('fingerprint')
        if not stored_fingerprint:
            return jsonify({'success': False, 'error': 'No fingerprint data found'})
            
        # Check if a fingerprint image file is included
        if 'fingerprint_image' in request.files:
            fingerprint_file = request.files['fingerprint_image']
            
            # Get sequence information
            sequence_number = int(request.form.get('sequence_number', 1))
            
            # Import the verify module
            from FingerPrint.everify import verify_fingerprint
            
            # Verify the fingerprint
            try:
                user_id = session['user_id']
                result = verify_fingerprint(fingerprint_file, user_id, sequence_number)
                
                # Ensure we handle the returned data properly, with defaults for missing values
                total_prints = result.get('total_fingerprints', 1)
                current_seq = result.get('current_sequence', sequence_number)
                is_last = result.get('is_last', current_seq >= total_prints)
                
                return jsonify({
                    'success': bool(result.get('success', False)),
                    'similarity': float(result.get('similarity_percent', 0.0)),
                    'threshold': float(result.get('threshold_percent', 0.0)),
                    'message': str(result.get('message', '')),
                    'total_fingerprints': total_prints,
                    'current_sequence': current_seq,
                    'is_last': is_last
                })
            except Exception as e:
                print(f"Error verifying fingerprint: {str(e)}")
                traceback.print_exc()
                return jsonify({
                    'success': False, 
                    'error': f"Fingerprint verification failed: {str(e)}"
                })
        
        # If no file is uploaded, return error
        return jsonify({'success': False, 'error': 'No fingerprint image provided'})
    
    elif biometric_type == 'eyeblink':
        try:
            # Get the stored eye blink pattern
            stored_pattern = user['biometrics'].get('eyeblink')
            if not stored_pattern:
                return jsonify({'success': False, 'error': 'No eye blink pattern stored'})
            
            # Kill any existing EBR processes to avoid duplicate windows
            os.system("pkill -f 'python.*EBR.py'")
            
            # Use subprocess to run EBR.py with --verify flag and the stored pattern
            result = subprocess.run(
                ['python', os.path.join(BASE_DIR, 'EyeBlink', 'EBR.py'), 
                 '--verify', stored_pattern, '--timeout', '60'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Process output
            output = result.stdout.strip()
            print(f"EBR.py verification output: {output}")
            
            # Parse verification result from output
            if "verified': True" in output or "Password verification successful" in output:
                return jsonify({'success': True})
            
            # Try JSON format
            match = re.search(r'{.*}', output)
            if match:
                try:
                    json_str = match.group(0).replace("'", '"')
                    result_json = json.loads(json_str)
                    if result_json.get('verified', False):
                        return jsonify({'success': True})
                except Exception as e:
                    print(f"JSON parsing error: {e}")
            
            # If we get here, verification failed
            return jsonify({
                'success': False, 
                'error': 'Eye blink pattern verification failed'
            })
                
        except Exception as e:
            print(f"Error in EBR verification: {str(e)}")
            traceback.print_exc()  # Add this to get full stack trace
            return jsonify({
                'success': False, 
                'error': f'Error during eye blink verification: {str(e)}'
            })
    
    elif biometric_type == 'neck_movement':
        try:
            stored_pattern = user['biometrics'].get('neck_movement')
            if not stored_pattern:
                return jsonify({'success': False, 'error': 'No neck movement pattern stored'})
            
            # Kill any existing neckmov processes
            os.system("pkill -f 'python.*neckmov.py'")
            
            # Run neckmov.py with --verify flag and single-window flag
            result = subprocess.run(
                ['python', os.path.join(BASE_DIR, 'NeckMovement', 'neckmov.py'), 
                 '--verify', stored_pattern, '--timeout', '60', '--single-window'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Process output
            output = result.stdout.strip()
            print(f"neckmov.py verification output: {output}")
            
            verification_successful = False
            if "verified': True" in output or "Verification successful" in output:
                verification_successful = True
            else:
                # Try JSON format
                match = re.search(r'{.*}', output)
                if match:
                    try:
                        json_str = match.group(0).replace("'", '"')
                        result_json = json.loads(json_str)
                        if result_json.get('verified', False):
                            verification_successful = True
                    except Exception as e:
                        print(f"JSON parsing error: {e}")
            
            if verification_successful:
                return jsonify({'success': True})
            else:
                return jsonify({
                    'success': False, 
                    'message': 'Neck movement verification failed'
                })
        except Exception as e:
            print(f"Error in neck movement verification: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'message': f'Neck movement verification error: {str(e)}'
            })
    
    elif biometric_type == 'voice':
        try:
            # Get the stored voice model path from user biometrics
            voice_model_path = user['biometrics'].get('voice')
            if not voice_model_path or not os.path.exists(voice_model_path):
                return jsonify({'success': False, 'error': 'No voice biometric data found'})
            
            # Get the user directory from the model path
            user_voice_dir = os.path.dirname(voice_model_path)
            
            # Set environment variable for the voice modules
            print(f"Setting VOICE_FOLDER to: {user_voice_dir}")
            os.environ['VOICE_FOLDER'] = user_voice_dir
            
            try:
                # Import the verification function directly
                from Voice.voicevalidate import verify_voice_biometrics
                
                # Capture output with StringIO
                original_stdout = sys.stdout
                output_capture = io.StringIO()
                sys.stdout = output_capture
                
                # Run the verification and get the boolean result
                verification_result = verify_voice_biometrics()
                
                # Convert numpy bool_ to regular bool if needed
                if hasattr(verification_result, 'item'):
                    verification_success = bool(verification_result.item())
                else:
                    verification_success = bool(verification_result)
                
                # Get and parse the captured output
                output = output_capture.getvalue()
                sys.stdout = original_stdout  # Restore stdout
                
                # Extract scores from the output if available
                avg_score_match = re.search(r'Average score: ([-\d.]+)', output)
                threshold_match = re.search(r'Threshold: ([-\d.]+)', output)
                
                avg_score = float(avg_score_match.group(1)) if avg_score_match else None
                threshold = float(threshold_match.group(1)) if threshold_match else None
                
                return jsonify({
                    'success': verification_success,
                    'output': output,
                    'output_lines': output.strip().split('\n') if output else [],
                    'avg_score': avg_score,
                    'threshold': threshold
                })
            except Exception as e:
                # Make sure stdout is restored
                if 'original_stdout' in locals():
                    sys.stdout = original_stdout
                print(f"Error in voice verification function: {str(e)}")
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        except Exception as e:
            print(f"Error in voice verification: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'error': f'Voice verification error: {str(e)}'
            })
    
    elif biometric_type == 'face_detection':
        try:
            # Get user ID for authentication
            user_id = session['user_id']
            user_name = f"user_{user_id}"
            
            # Get the stored biometric data path 
            face_model_path = user['biometrics'].get('face_detection')
            if not face_model_path or not os.path.exists(face_model_path):
                return jsonify({'success': False, 'error': 'No face biometric data found'})
            
            # Important: Set paths correctly
            face_rec_dir = os.path.join(BASE_DIR, 'FaceRecognition')
            # Use absolute path for model weights
            model_weights_path = os.path.join(face_rec_dir, 'facenet_keras_weights.h5')
            
            if not os.path.exists(model_weights_path):
                return jsonify({'success': False, 'error': f'Model weights file not found at: {model_weights_path}'})
            
            # Important: Kill any existing face detection processes first
            os.system("pkill -f 'python.*detectface.py'")
            
            # Create a temporary file to store results
            result_file = os.path.join(TEMP_UPLOAD_FOLDER, f"face_auth_result_{user_id}_{uuid.uuid4()}.txt")
            
            print(f"Starting face verification for user: {user_name}")
            print(f"Model path: {face_model_path}")
            print(f"Weights path: {model_weights_path}")
            
            # Change directory to the FaceRecognition folder to ensure relative paths work
            current_dir = os.getcwd()
            os.chdir(face_rec_dir)
            
            # Run detectface.py with its working directory set to FaceRecognition
            process = subprocess.Popen([
                sys.executable,
                'detectface.py',
                '--authenticate',
                '--person', user_name,
                '--encodings', face_model_path,
                '--model', model_weights_path,
                '--result-file', result_file
            ])
            
            # Restore original directory
            os.chdir(current_dir)
            
            # Start polling for the verification result
            return jsonify({
                'success': True,
                'message': 'Face verification process started',
                'result_file': result_file
            })
        except Exception as e:
            print(f"Error during face verification: {e}")
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'error': f'Error launching face verification: {str(e)}'
            })
    
    else:
        return jsonify({
            'success': False, 
            'error': f'Invalid biometric type: {biometric_type}'
        })

# Grant access to hospital dashboard
@app.route('/grant_hospital_access', methods=['POST'])
def grant_hospital_access():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    # Set session flag to indicate hospital access is granted
    session['hospital_access_granted'] = True
    
    return jsonify({'success': True})

# Hospital dashboard route
@app.route('/hospital_dashboard')
def hospital_dashboard():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    
    # Check if user is verified for hospital dashboard access
    if not session.get('hospital_access_granted'):
        flash("You need to verify your identity first", "warning")
        return redirect(url_for('access_hospital_dashboard'))
    
    # Get user for displaying personalized information
    user = get_user_by_id(session['user_id'])
    
    return render_template('hospital_dashboard.html', user=user)

# Health Assistant route
@app.route('/health_assistant')
def health_assistant():
    if 'user_id' not in session:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))
    
    # Check if user is verified for hospital dashboard access
    if not session.get('hospital_access_granted'):
        flash("Please access the hospital dashboard first", "warning")
        return redirect(url_for('access_hospital_dashboard'))
    
    # Start the RAG app in the background if it's not already running
    start_rag_app()
    
    # Return the template that will redirect to the RAG app
    return render_template('health_assistant.html', user=get_user_by_id(session['user_id']))

# Add this cleanup function to ensure the RAG app is terminated when the main app stops
@app.teardown_appcontext
def cleanup_rag_app(exception=None):
    global rag_app_process
    if rag_app_process is not None:
        try:
            rag_app_process.terminate()
            rag_app_process.wait(timeout=5)
        except:
            if rag_app_process.poll() is None:
                rag_app_process.kill()

# Add these proxy routes to your main Flask application

# Proxy route for the RAG health check
@app.route('/proxy/rag/health', methods=['GET'])
def proxy_rag_health():
    if not session.get('hospital_access_granted'):
        return jsonify({'error': 'Not authorized'}), 403
    
    try:
        response = requests.get(f"http://127.0.0.1:{RAG_APP_PORT}/api/health", timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e),
            'documents': {'available': False, 'count': 0}
        })

# Proxy route for document uploads
@app.route('/proxy/rag/upload', methods=['POST'])
def proxy_rag_upload():
    if not session.get('hospital_access_granted'):
        return jsonify({'error': 'Not authorized'}), 403
    
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        # Forward the file to the RAG app
        files = {'file': (file.filename, file.read(), file.content_type)}
        response = requests.post(f"http://127.0.0.1:{RAG_APP_PORT}/upload", files=files)
        
        # Return the same status code and response from the RAG app
        return response.text, response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Proxy route for database reset
@app.route('/proxy/rag/reset', methods=['POST'])
def proxy_rag_reset():
    if not session.get('hospital_access_granted'):
        return jsonify({'error': 'Not authorized'}), 403
    
    try:
        rebuild = request.form.get('rebuild', 'false')
        response = requests.post(
            f"http://127.0.0.1:{RAG_APP_PORT}/reset",
            data={'rebuild': rebuild}
        )
        return response.text, response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Proxy route for medical queries
@app.route('/proxy/rag/medical-query', methods=['POST'])
def proxy_rag_medical_query():
    if not session.get('hospital_access_granted'):
        return jsonify({'error': 'Not authorized'}), 403
    
    try:
        data = request.json
        response = requests.post(
            f"http://127.0.0.1:{RAG_APP_PORT}/api/medical-query",
            json=data
        )
        return response.json(), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

migrate_to_encrypted_data()
cleanup_temp_directories()  # Add this line

if __name__ == '__main__':
    app.run(debug=True)