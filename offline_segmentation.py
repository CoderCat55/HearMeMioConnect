import os
import numpy as np
from model1 import RestDetector

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "calibration_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

# Model 1 Parameters
WINDOW_SIZE = 20
THRESHOLD_FACTOR = 6.0  # Artırıldı: Gürültüyü hareket sanmaması için
MIN_DURATION = 20
PADDING = 0             # Sıfırlandı: Başta ve sonda rest istemiyoruz
# ==============================================================================

def load_npy(path):
    try:
        data = np.load(path, allow_pickle=True)
        # Handle different shapes/formats of saved data
        if data.ndim == 2: return data
        # If it's a list of [timestamp, data]
        if len(data.shape) > 1 and len(data[0]) == 2:
             return np.vstack([x[1] for x in data])
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    print("=== Offline Segmentation Started ===")
    
    # 1. Initialize Model 1
    detector = RestDetector(WINDOW_SIZE, THRESHOLD_FACTOR, MIN_DURATION, PADDING)
    
    # 2. Find Rest Data for Calibration
    rest_data = []
    print(f"Scanning {CALIB_DIR} for 'rest' files...")
    for root, dirs, files in os.walk(CALIB_DIR):
        for file in files:
            if "rest" in file.lower() and file.endswith(".npy"):
                path = os.path.join(root, file)
                data = load_npy(path)
                if data is not None:
                    rest_data.append(data)
                    print(f"  Loaded rest file: {file}")

    if not rest_data:
        print("ERROR: No rest files found. Cannot calibrate Model 1.")
        return

    # 3. Train Model 1 (Calculate Threshold)
    if not detector.fit(rest_data):
        return
    
    # 4. Process All Gesture Files
    print("\nProcessing gesture files...")
    count = 0
    for root, dirs, files in os.walk(CALIB_DIR):
        for file in files:
            if file.endswith(".npy") and "rest" not in file.lower():
                input_path = os.path.join(root, file)
                
                # Determine output path (preserve folder structure)
                rel_path = os.path.relpath(root, CALIB_DIR)
                target_dir = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                
                # Load and Segment
                data = load_npy(input_path)
                if data is None: continue
                
                segments = detector.segment(data)
                
                # Save Segments
                base_name = os.path.splitext(file)[0]
                if not segments:
                    print(f"  Skipping {file} (No activity detected)")
                    continue
                    
                for i, seg in enumerate(segments):
                    suffix = f"_seg{i}" if len(segments) > 1 else ""
                    save_name = f"{base_name}{suffix}.npy"
                    np.save(os.path.join(target_dir, save_name), seg)
                    
                print(f"  Processed {file} -> {len(segments)} segments saved.")
                count += 1
                
    print(f"\n=== Done. Processed {count} files. ===")

if __name__ == "__main__":
    main()