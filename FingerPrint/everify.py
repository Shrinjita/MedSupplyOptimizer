import os
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import json

# Get base directory (where the app is running)
BASE_DIR = os.getcwd()
FOLDER_PATH = os.path.join(BASE_DIR, "FingerPrint")

def verify_fingerprint(uploaded_file, user_id, sequence_number=1):
    """Verify the uploaded fingerprint against the stored fingerprint."""
    # Get path to stored fingerprint
    user_dir = os.path.join(FOLDER_PATH, f"user_{user_id}")
    stored_path = os.path.join(user_dir, f"fingerprint_{sequence_number}.png")
    metadata_path = os.path.join(user_dir, "fingerprint_metadata.json")

    # Check if stored fingerprint exists
    if not os.path.exists(stored_path):
        return {
            "success": False, 
            "error": f"No fingerprint #{sequence_number} enrolled for this user."
        }
        
    # Load metadata to get total fingerprints
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        total_fingerprints = metadata.get("total_fingerprints", 1)
    except:
        total_fingerprints = 1

    # Load the stored fingerprint
    stored_img = cv2.imread(stored_path, cv2.IMREAD_GRAYSCALE)
    if stored_img is None:
        return {
            "success": False, 
            "error": f"Failed to load stored fingerprint #{sequence_number}."
        }

    # Load the uploaded fingerprint
    uploaded_img = Image.open(uploaded_file).convert('L')
    uploaded_img = np.array(uploaded_img)

    # Resize the uploaded image to match the stored image dimensions
    uploaded_img = cv2.resize(uploaded_img, (stored_img.shape[1], stored_img.shape[0]))

    # Compare the fingerprints using SSIM
    similarity = ssim(stored_img, uploaded_img)

    # Define a threshold for fingerprint matching
    threshold = 0.95
    success = similarity >= threshold

    return {
        "success": success,
        "similarity": similarity,
        "similarity_percent": round(similarity * 100, 2),
        "threshold": threshold,
        "threshold_percent": round(threshold * 100),
        "total_fingerprints": total_fingerprints,
        "current_sequence": sequence_number,
        "is_last": sequence_number >= total_fingerprints,
        "message": f"Fingerprint #{sequence_number} match successful." if success 
                  else f"Fingerprint #{sequence_number} match failed."
    }

def get_fingerprint_metadata(user_id):
    """Get metadata about enrolled fingerprints."""
    user_dir = os.path.join(FOLDER_PATH, f"user_{user_id}")
    metadata_path = os.path.join(user_dir, "fingerprint_metadata.json")
    
    if not os.path.exists(metadata_path):
        return {"total_fingerprints": 0}
        
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except:
        return {"total_fingerprints": 0}