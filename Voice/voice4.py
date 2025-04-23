import os
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import time
import shutil  
from scipy.signal import butter, filtfilt 

PHRASE = "I let the positive overrun the negative"
SAMPLE_RATE = 48000
DURATION = 5
TEST_SAMPLES = 2
VOICE_FOLDER = "user" 
VOICE_MODEL_FILE = os.path.join(VOICE_FOLDER, "voice_model.pkl")
VERIFICATION_THRESHOLD = -310.0  


def record_audio(output_file, duration=DURATION, fs=SAMPLE_RATE):
   
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
    
    
    sf.write(output_file, audio_data, fs)
    print(f"✅ Saved to {output_file}")
    
    return output_file


def reduce_noise(y, sr):
  
    cutoff = 80 
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    order = 4
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def extract_enhanced_features(file_path, n_mfcc=20):

    y, sr = librosa.load(file_path, sr=None)
    
 
    y = reduce_noise(y, sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
   
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    combined_features = np.vstack([
        np.mean(mfcc.T, axis=0),     
        np.std(mfcc.T, axis=0),      
        np.mean(delta_mfcc.T, axis=0),  
        np.mean(delta2_mfcc.T, axis=0) 
    ])
    
    return combined_features.flatten()


def has_voice_biometrics():
    return os.path.exists(VOICE_MODEL_FILE)

def cleanup_test_files(test_dir):

    print("\nCleaning up test recordings... ", end="", flush=True)
    try:
       
        files_removed = 0
        for file in os.listdir(test_dir):
            if file.endswith(".wav"):
                os.remove(os.path.join(test_dir, file))
                files_removed += 1
                
        print(f"DONE! (Removed {files_removed} files)")
    except Exception as e:
        print(f"FAILED! ({str(e)})")


def verify_voice_biometrics():
    print("\n=== VOICE VERIFICATION ===")

    if not has_voice_biometrics():
        print("❌ No voice biometric data found. Please register your voice first using voice3.py")
        return False
    
# load model
    print("Loading voice model... ", end="", flush=True)
    try:
        with open(VOICE_MODEL_FILE, "rb") as model_file:
            model_data = pickle.load(model_file)
            
        if len(model_data) == 4:
            gmm, scaler, stored_phrase, _ = model_data  # Ignore saved threshold
        elif len(model_data) == 3:
            gmm, scaler, stored_phrase = model_data
        else:
          
            gmm, scaler = model_data[:2]
            stored_phrase = PHRASE
        
     
        threshold = VERIFICATION_THRESHOLD
        print("DONE!")
            
    except Exception as e:
        print("FAILED!")
        print(f"❌ Error loading voice model: {str(e)}")
        return False
    

    test_dir = os.path.join(VOICE_FOLDER, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"You will say: \"{stored_phrase}\" {TEST_SAMPLES} times for verification")
    
    scores = []

    for i in range(1, TEST_SAMPLES + 1):
        print(f"\nTest sample {i}/{TEST_SAMPLES}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(test_dir, f"test_{i}_{timestamp}.wav")
        
        try:
            record_audio(output_file)
            
         
            print("Analyzing voice... ", end="", flush=True)
            
       
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
        return False
    

    avg_score = sum(scores) / len(scores)
    
    print("\nCalculating final result...")
    time.sleep(1) 
    
    print(f"Average score: {avg_score:.2f}")
    print(f"Threshold: {threshold}")
    
 
    result = avg_score >= threshold
    
    return result


if __name__ == "__main__":
    print("=== Voice Biometrics Verification ===")
    
    if not has_voice_biometrics():
        print("❌ No voice biometric data found.")
        print("Please register your voice first by running voice3.py")
    else:
        print("Voice biometric data found. Starting verification...")
        success = verify_voice_biometrics()
        
        if success:
            print("\n✅ VERIFICATION SUCCESSFUL")
            print("Access granted.")
        else:
            print("\n❌ VERIFICATION FAILED")
            print("Access denied.")


        # Clean up test recordings
        test_dir = os.path.join(VOICE_FOLDER, "test")
        if os.path.exists(test_dir):
            cleanup_test_files(test_dir)    