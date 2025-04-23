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

# Config
PHRASE = "I let the positive overrun the negative"
SAMPLE_RATE = 48000
DURATION = 5 
TRAIN_SAMPLES = 4
VOICE_FOLDER = "user"  
VOICE_MODEL_FILE = os.path.join(VOICE_FOLDER, "voice_model.pkl")


os.makedirs(VOICE_FOLDER, exist_ok=True)


def reduce_noise(y, sr):
  
    cutoff = 80
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    order = 4
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered


def augment_audio(audio_file, output_file, augmentation_type='pitch'):

    y, sr = librosa.load(audio_file, sr=None)
    

    y = reduce_noise(y, sr)
    
    if augmentation_type == 'pitch':

        pitch_shift = random.uniform(-0.05, 0.05) 
        y_augmented = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
    elif augmentation_type == 'speed':
        
        speed_factor = random.uniform(0.995, 1.005)
        y_augmented = librosa.effects.time_stretch(y, rate=speed_factor)
    else:
      
        noise_level = 0.0002
        noise = np.random.normal(0, noise_level, len(y))
        y_augmented = y + noise
  
    sf.write(output_file, y_augmented, sr)
    return output_file



def record_audio(output_file, duration=DURATION, fs=SAMPLE_RATE):

    print(f"\nPlease say: \"{PHRASE}\"")
    print("Recording will start in...")
    
  
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
    
  
    audio_data = audio_data.flatten()
    audio_data = reduce_noise(audio_data, fs)
    

    sf.write(output_file, audio_data, fs)
    print(f"✅ Saved to {output_file}")
    
    return output_file

# Extract enhanced features
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

def register_voice_biometrics(train_dir=os.path.join(VOICE_FOLDER, "train")):
   
    print("\n=== VOICE BIOMETRICS REGISTRATION ===")
    print(f"You will record the phrase \"{PHRASE}\" {TRAIN_SAMPLES} times")
    
 
    os.makedirs(train_dir, exist_ok=True)
    
    features = []
    recorded_files = []

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
    

    if len(recorded_files) >= 2:
        print("\nCreating augmented samples to enhance voice profile...")

        for i, recording in enumerate(recorded_files):
            if i%2==0:
                aug_file = os.path.join(train_dir, f"train_aug_pitch_{i}_{timestamp}.wav")
                augment_audio(recording, aug_file, 'pitch')
                user_features = extract_enhanced_features(aug_file)
                features.append(user_features)
                print(f"✅ Created augmented sample for recording {i+1} (pitch shift)")

            else: 
                aug_file = os.path.join(train_dir, f"train_aug_speed_{i}_{timestamp}.wav")
                augment_audio(recording, aug_file, 'speed')
                user_features = extract_enhanced_features(aug_file)
                features.append(user_features)
                print(f"✅ Created augmented sample for recording {i+1} (speed modification)")    
            




      
        
  
        
    
    if len(features) < 4:
        print("❌ Not enough valid recordings to create a voice profile")
        return False
        
    features = np.array(features)
    print(f"\nTraining with {len(features)} audio samples, feature dimension: {features.shape}")
    

    print("Training model... ", end="", flush=True)
    

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    

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
        
  
        with open(VOICE_MODEL_FILE, "wb") as f:
            pickle.dump((gmm, scaler, PHRASE), f)
        print("DONE!")
        print(f"✅ Voice biometric data saved successfully to {VOICE_MODEL_FILE}")
        return True
    except Exception as e:
        print("FAILED!")
        print(f"❌ Error training model: {str(e)}")
        return False


def has_voice_biometrics():
  
    return os.path.exists(VOICE_MODEL_FILE)


if __name__ == "__main__":
    print("=== Voice Biometrics Registration ===")
    
    if has_voice_biometrics():
        print("Voice biometric data already exists.")
        choice = input("Do you want to re-register your voice? (y/n): ")
        if choice.lower() != 'y':
            print("Registration cancelled. Existing voice data retained.")
            exit()
        print("Proceeding with re-registration...")
    else:
        print("No voice biometric data found. Creating new registration.")
    

    success = register_voice_biometrics()
    
    if success:
        print("\nVoice registration completed successfully.")
        print(f"Model saved to {VOICE_MODEL_FILE}")
    else:
        print("\nVoice registration failed. Please try again.")