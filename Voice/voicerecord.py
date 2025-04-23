import os
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import random
from scipy.signal import butter, filtfilt

# Configuration
PHRASE = "I let the positive overrun the negative"
SAMPLE_RATE = 48000
DURATION = 5 
TRAIN_SAMPLES = 4  # Using 4 samples instead of 6

# DO NOT create a default "user" folder - only use the environment variable
if not os.environ.get('VOICE_FOLDER'):
    print("WARNING: VOICE_FOLDER environment variable not set. Voice operations may fail.")
    # We'll set this to a non-existent path so it won't create unwanted folders
    VOICE_FOLDER = "__VOICE_FOLDER_NOT_SET__"
else:
    VOICE_FOLDER = os.environ.get('VOICE_FOLDER')
    # Only create the directory if the environment variable is properly set
    os.makedirs(VOICE_FOLDER, exist_ok=True)

VOICE_MODEL_FILE = os.path.join(VOICE_FOLDER, "voice_model.pkl")

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

# Update the augment_audio function to confirm file saving
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
    
    # Ensure y_augmented is properly formatted for writing
    if len(y_augmented.shape) == 1:
        y_augmented = y_augmented.reshape(-1, 1)
    
    # Save the file
    sf.write(output_file, y_augmented, sr)
    
    # Confirm file was saved
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✅ Augmented file saved: {os.path.basename(output_file)} ({file_size} bytes)")
    else:
        print(f"❌ Failed to save augmented file: {output_file}")
    
    return output_file

# Terminal-based audio recording function
def record_audio(output_file, duration=DURATION, fs=SAMPLE_RATE):
    """Record audio from terminal and save to file with noise reduction"""
    # First make sure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nPlease say: \"{PHRASE}\"")
    print("Recording will start in...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording NOW! 🎤")
    
    # Record audio
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    
    # Show progress bar
    for i in range(duration):
        time.sleep(1)
        bars = "█" * (i + 1) + "░" * (duration - i - 1)
        print(f"\rRecording: {bars} {i+1}/{duration}s", end="", flush=True)
    
    # Wait for recording to complete
    sd.wait()
    print("\n✅ Recording complete!")
    
    # Apply noise reduction
    audio_data = audio_data.flatten()
    audio_data = reduce_noise(audio_data, fs)
    
    # Save audio to file
    sf.write(output_file, audio_data, fs)
    print(f"✅ Saved to {output_file}")
    
    return output_file

# Extract enhanced features
def extract_enhanced_features(file_path, n_mfcc=20):
    """Extract acoustic features from audio file with noise reduction"""
    y, sr = librosa.load(file_path, sr=None)
    
    # Apply noise reduction
    y = reduce_noise(y, sr)
    
    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Add delta and delta-delta features (velocity and acceleration)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Combine features into a single vector
    combined_features = np.vstack([
        np.mean(mfcc.T, axis=0),      # Mean of MFCCs
        np.std(mfcc.T, axis=0),       # Standard deviation of MFCCs
        np.mean(delta_mfcc.T, axis=0),  # Mean of deltas
        np.mean(delta2_mfcc.T, axis=0)  # Mean of delta-deltas
    ])
    
    return combined_features.flatten()

# Register voice biometrics function
def register_voice_biometrics(train_dir=None):
    """Record voice samples and create a voice biometric model with augmentation"""
    if not train_dir:
        if not os.environ.get('VOICE_FOLDER'):
            print("❌ VOICE_FOLDER environment variable not set")
            return False
        train_dir = os.path.join(VOICE_FOLDER, "train")
    
    print("\n=== VOICE BIOMETRICS REGISTRATION ===")
    print(f"Using voice directory: {VOICE_FOLDER}")
    print(f"You will record the phrase \"{PHRASE}\" {TRAIN_SAMPLES} times")
    
    # Ensure directory exists
    os.makedirs(train_dir, exist_ok=True)
    
    features = []
    recorded_files = []
    
    # Record training samples
    for i in range(1, TRAIN_SAMPLES + 1):
        print(f"\nTraining sample {i}/{TRAIN_SAMPLES}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(train_dir, f"train_{i}_{timestamp}.wav")
        
        try:
            record_audio(output_file)
            recorded_files.append(output_file)
            
            # Extract features
            user_features = extract_enhanced_features(output_file)
            features.append(user_features)
            print(f"Processed: {os.path.basename(output_file)}")
        except Exception as e:
            print(f"Error processing recording {i}: {str(e)}")
            print("Please try again.")
            i -= 1  # Retry this recording
            continue
    
    # Create augmented samples to enhance model with fewer recordings
    if len(recorded_files) >= 2:
        print("\nCreating augmented samples to enhance voice profile...")

        for i, recording in enumerate(recorded_files):
            # Create a new timestamp for each augmented file to avoid conflicts
            aug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            if i % 2 == 0:
                # Add pitch-shifted version
                aug_file = os.path.join(train_dir, f"train_aug_pitch_{i}_{aug_timestamp}.wav")
                augment_audio(recording, aug_file, 'pitch')
                
                # Check if file exists before extracting features
                if os.path.exists(aug_file):
                    user_features = extract_enhanced_features(aug_file)
                    features.append(user_features)
                    print(f"✅ Added augmented sample features for recording {i+1} (pitch shift)")
                else:
                    print(f"⚠️ Could not process augmented file - not found: {aug_file}")
            else: 
                # Add speed-modified version
                aug_file = os.path.join(train_dir, f"train_aug_speed_{i}_{aug_timestamp}.wav")
                augment_audio(recording, aug_file, 'speed')
                
                # Check if file exists before extracting features
                if os.path.exists(aug_file):
                    user_features = extract_enhanced_features(aug_file)
                    features.append(user_features)
                    print(f"✅ Added augmented sample features for recording {i+1} (speed modification)")
                else:
                    print(f"⚠️ Could not process augmented file - not found: {aug_file}")
    
    if len(features) < 4:
        print("❌ Not enough valid recordings to create a voice profile")
        return False
        
    features = np.array(features)
    print(f"\nTraining with {len(features)} audio samples, feature dimension: {features.shape}")
    
    # Display a loading indicator
    print("Training model... ", end="", flush=True)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create GMM with proper regularization
    gmm = GaussianMixture(
        n_components=min(2, len(features)-1),  # Adaptive number of components
        covariance_type='diag',   # Diagonal is more stable than full
        n_init=10,
        reg_covar=1.0,
        random_state=42,
        max_iter=300
    )
    
    try:
        gmm.fit(features_scaled)
        
        # Save both the model and scaler together in the fixed location
        with open(VOICE_MODEL_FILE, "wb") as f:
            pickle.dump((gmm, scaler, PHRASE), f)
        print("DONE!")
        print(f"✅ Voice biometric data saved successfully to {VOICE_MODEL_FILE}")
        return True
    except Exception as e:
        print("FAILED!")
        print(f"❌ Error training model: {str(e)}")
        return False

# Check if voice biometrics exist
def has_voice_biometrics():
    """Check if voice biometric data already exists"""
    return os.path.exists(VOICE_MODEL_FILE)

# Main Execution - Registration Only
if __name__ == "__main__":
    print("=== Voice Biometrics Registration ===")
    
    if not os.environ.get('VOICE_FOLDER'):
        print("ERROR: VOICE_FOLDER environment variable must be set")
        print("Example: export VOICE_FOLDER=/path/to/voice/folder")
        exit(1)
        
    if has_voice_biometrics():
        print("Voice biometric data already exists.")
        choice = input("Do you want to re-register your voice? (y/n): ")
        if choice.lower() != 'y':
            print("Registration cancelled. Existing voice data retained.")
            exit()
        print("Proceeding with re-registration...")
    else:
        print("No voice biometric data found. Creating new registration.")
    
    # Proceed with registration
    success = register_voice_biometrics()
    
    if success:
        print("\nVoice registration completed successfully.")
        print(f"Model saved to {VOICE_MODEL_FILE}")
    else:
        print("\nVoice registration failed. Please try again.")