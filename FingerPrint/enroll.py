import os
from PIL import Image
import json

# Get base directory (where the app is running)
BASE_DIR = os.getcwd()
FOLDER_PATH = os.path.join(BASE_DIR, "FingerPrint")

def save_fingerprint(image_file, user_id, sequence_number=1, total_prints=1):
    """Save the uploaded fingerprint image with sequence information."""
    # Create a directory for the user's fingerprints
    user_dir = os.path.join(FOLDER_PATH, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    # Save the fingerprint image with sequence number
    fingerprint_path = os.path.join(user_dir, f"fingerprint_{sequence_number}.png")
    
    # Convert to grayscale for better matching
    image = Image.open(image_file).convert('L')
    image.save(fingerprint_path)
    
    # Store sequence metadata
    metadata = {
        "total_fingerprints": total_prints,
        "last_updated": os.path.getmtime(fingerprint_path)
    }
    
    # Save metadata
    metadata_path = os.path.join(user_dir, "fingerprint_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    print(f"✅ Fingerprint #{sequence_number} enrolled successfully at {fingerprint_path}!")
    
    # Return the directory path rather than individual file
    return user_dir

if __name__ == "__main__":
    user_id = input("Enter your user ID: ").strip()
    total_prints = int(input("How many fingerprints do you want to enroll (1-5)? "))
    
    if total_prints < 1 or total_prints > 5:
        print("Number of fingerprints must be between 1 and 5.")
        exit()
        
    for seq in range(1, total_prints + 1):
        image_file = input(f"Enter the path to fingerprint #{seq}: ").strip()
        if os.path.exists(image_file):
            fingerprint_dir = save_fingerprint(image_file, user_id, seq, total_prints)
            print(f"✅ Fingerprint #{seq} saved successfully!")
        else:
            print("Invalid image path. Please try again.")
            seq -= 1  # Retry this sequence number