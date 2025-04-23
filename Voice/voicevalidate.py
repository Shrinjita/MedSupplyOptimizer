import os
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import time
import shutil  # For directory cleanup
from scipy.signal import butter, filtfilt

# Configuration constants
PHRASE = "I let the positive overrun the negative"
SAMPLE_RATE = 48000
DURATION = 5
TEST_SAMPLES = 2

# Apply high-pass filter to reduce low frequency noise
def reduce_noise(y, sr):
    """Apply high-pass filter to reduce low frequency noise"""
    cutoff = 80  # Cut off frequency below 80Hz
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    order = 4
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered

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
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Add delta and delta-delta features
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Combine features
    combined_features = np.vstack([
        np.mean(mfcc.T, axis=0),     
        np.std(mfcc.T, axis=0),      
        np.mean(delta_mfcc.T, axis=0),  
        np.mean(delta2_mfcc.T, axis=0) 
    ])
    
    return combined_features.flatten()

# Check if voice biometrics exist
def has_voice_biometrics():
    """Check if voice model exists at configured path"""
    # Get VOICE_FOLDER at function call time
    voice_folder = os.environ.get('VOICE_FOLDER')
    if not voice_folder:
        print("ERROR: VOICE_FOLDER environment variable not set")
        return False
    
    voice_model_file = os.path.join(voice_folder, "voice_model.pkl")
    return os.path.exists(voice_model_file)

# Cleanup test files
def cleanup_test_files(test_dir):
    """Remove all wav files from test directory"""
    print("\nCleaning up test recordings... ", end="", flush=True)
    try:
        # Method 1: Delete individual files
        files_removed = 0
        for file in os.listdir(test_dir):
            if file.endswith(".wav"):
                os.remove(os.path.join(test_dir, file))
                files_removed += 1
                
        print(f"DONE! (Removed {files_removed} files)")
    except Exception as e:
        print(f"FAILED! ({str(e)})")

# Verify voice biometrics
def verify_voice_biometrics():
    """Verify user using voice biometrics"""
    # Get VOICE_FOLDER at verification time from environment
    voice_folder = os.environ.get('VOICE_FOLDER')
    if not voice_folder:
        print("ERROR: VOICE_FOLDER environment variable not set")
        print("Please set it to the correct voice model directory before verification")
        return False
        
    print("\n=== VOICE VERIFICATION ===")
    print(f"Using voice directory: {voice_folder}")
    
    # Define model path based on current environment variable
    voice_model_file = os.path.join(voice_folder, "voice_model.pkl")
    
    # Check if model file exists
    if not os.path.exists(voice_model_file):
        print(f"❌ No voice biometric data found at {voice_model_file}")
        return False
    
    # Load model
    print("Loading voice model... ", end="", flush=True)
    try:
        with open(voice_model_file, "rb") as model_file:
            model_data = pickle.load(model_file)
            
        if len(model_data) == 3:
            gmm, scaler, stored_phrase = model_data
        else:
            # Handle older model formats
            gmm, scaler = model_data[:2]
            stored_phrase = PHRASE
        print("DONE!")
            
    except Exception as e:
        print("FAILED!")
        print(f"❌ Error loading voice model: {str(e)}")
        return False
    
    # Ensure test directory exists
    test_dir = os.path.join(voice_folder, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"You will say: \"{stored_phrase}\" {TEST_SAMPLES} times for verification")
    
    scores = []
    
    # Check if test files already exist (reuse them if they do)
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    
    if test_files and len(test_files) >= TEST_SAMPLES:
        print(f"Found {len(test_files)} existing test files - using those instead of recording")
        
        # Use existing test files
        for i, test_file in enumerate(test_files[:TEST_SAMPLES]):
            file_path = os.path.join(test_dir, test_file)
            
            try:
                # Extract features and get score
                print(f"Processing test file {i+1}: {test_file}")
                test_features = extract_enhanced_features(file_path).reshape(1, -1)
                test_features_scaled = scaler.transform(test_features)
                score = gmm.score(test_features_scaled)
                scores.append(score)
                print(f"🔍 Sample score: {score:.2f}")
            except Exception as e:
                print(f"Error processing test file {test_file}: {str(e)}")
    else:
        # Record new test samples
        for i in range(1, TEST_SAMPLES + 1):
            print(f"\nTest sample {i}/{TEST_SAMPLES}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(test_dir, f"test_{i}_{timestamp}.wav")
            
            try:
                record_audio(output_file)
                
                # Display analyzing message
                print("Analyzing voice... ", end="", flush=True)
                
                # Extract features and get score
                test_features = extract_enhanced_features(output_file).reshape(1, -1)
                test_features_scaled = scaler.transform(test_features)
                score = gmm.score(test_features_scaled)
                scores.append(score)
                print("DONE!")
                print(f"🔍 Sample score: {score:.2f}")
                
            except Exception as e:
                print("FAILED!")
                print(f"Error processing recording: {str(e)}")
    
    if not scores:
        print("❌ No valid recordings for verification")
        # Clean up even if verification failed
        cleanup_test_files(test_dir)
        return False
    
    # Make verification decision
    avg_score = sum(scores) / len(scores)
    threshold = -198.0  # Adjust based on testing
    
    print("\nCalculating final result...")
    print(f"Average score: {avg_score:.2f}")  # Ensure exact format for regex matching
    print(f"Threshold: {threshold:.1f}")  # Ensure exact format for regex matching
    
    # Determine result
    result = avg_score >= threshold
    
    # Clean up test files regardless of verification result
    cleanup_test_files(test_dir)
    
    if result:
        print("\n✅ VERIFICATION SUCCESSFUL")
        print("Voice pattern matched. Access granted.")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("Voice pattern did not match. Access denied.")
    
    return result

# Main execution - verification only
if __name__ == "__main__":
    print("=== Voice Biometrics Verification ===")
    
    if not os.environ.get('VOICE_FOLDER'):
        print("ERROR: VOICE_FOLDER environment variable must be set")
        print("Example: export VOICE_FOLDER=/path/to/voice/folder")
        exit(1)
    
    if not has_voice_biometrics():
        print("❌ No voice biometric data found.")
        print("Please register your voice first by running voicerecord.py")
        exit(1)
    
    voice_folder = os.environ.get('VOICE_FOLDER')
    voice_model_file = os.path.join(voice_folder, "voice_model.pkl")
    print(f"Voice biometric data found at {voice_model_file}. Starting verification...")
    success = verify_voice_biometrics()