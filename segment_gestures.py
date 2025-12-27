# segment_gestures.py
import numpy as np
import glob
import os
from rest_model import RestModel

def load_all_rest_data():
    """Load rest data from all participants (p1rest to p6rest)"""
    rest_data = []
    
    for participant_id in range(1, 7):
        folder = f'calibration_data/p{participant_id}rest'
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping...")
            continue
        
        files = glob.glob(f'{folder}/*.npy')
        for file in files:
            data = np.load(file)
            rest_data.append(data)
            print(f"Loaded rest data: {file} with shape {data.shape}")
    
    return rest_data

def segment_gesture_file(gesture_data, rest_model, min_segment_length_ms=100):
    """
    Segment a single gesture file by removing rest periods
    
    gesture_data: numpy array (time_steps, 34)
    rest_model: trained RestModel
    min_segment_length_ms: minimum length for valid segment
    
    Returns: list of segmented numpy arrays
    """
    min_samples = int((min_segment_length_ms / 1000) * rest_model.sampling_rate)
    
    # Slide 20ms window through data with 50% overlap
    num_windows = (len(gesture_data) - rest_model.samples_per_window) // rest_model.stride + 1
    
    # Predict rest/non-rest for each window
    predictions = []
    for i in range(num_windows):
        start_idx = i * rest_model.stride
        end_idx = start_idx + rest_model.samples_per_window
        window = gesture_data[start_idx:end_idx]
        
        features = rest_model.extract_features(window)
        is_rest = rest_model.is_rest(features)
        predictions.append((start_idx, end_idx, is_rest))
    
    # Find continuous non-rest segments
    segments = []
    current_segment_start = None
    
    for start_idx, end_idx, is_rest in predictions:
        if not is_rest:  # Non-rest window
            if current_segment_start is None:
                current_segment_start = start_idx
        else:  # Rest window
            if current_segment_start is not None:
                # End of non-rest segment
                segment_data = gesture_data[current_segment_start:end_idx]
                
                # Only keep if long enough
                if len(segment_data) >= min_samples:
                    segments.append(segment_data)
                
                current_segment_start = None
    
    # Handle case where file ends with non-rest
    if current_segment_start is not None:
        segment_data = gesture_data[current_segment_start:]
        if len(segment_data) >= min_samples:
            segments.append(segment_data)
    
    return segments

def segment_participant_data(participant_id, rest_model):
    """
    Process all gesture files for one participant
    
    participant_id: 1-6
    rest_model: trained RestModel
    """
    input_folder = f'calibration_data/p{participant_id}'
    output_folder = f'calibration_data/p{participant_id}new'
    
    if not os.path.exists(input_folder):
        print(f"Error: {input_folder} not found!")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    files = glob.glob(f'{input_folder}/*.npy')
    
    print(f"\nProcessing participant {participant_id}...")
    print(f"Found {len(files)} gesture files")
    
    total_segments = 0
    
    for file in files:
        # Extract gesture name from filename
        basename = os.path.basename(file)
        gesture_name = basename.split('_')[0]  # e.g., "fist_123456.npy" -> "fist"
        
        # Load gesture data
        gesture_data = np.load(file)
        print(f"  Processing {basename} (shape: {gesture_data.shape})...")
        
        # Segment
        segments = segment_gesture_file(gesture_data, rest_model)
        
        print(f"    -> Found {len(segments)} segments")
        
        # Save each segment
        for seg_idx, segment in enumerate(segments):
            output_filename = f"{output_folder}/{gesture_name}_seg{seg_idx}_{basename}"
            np.save(output_filename, segment)
            total_segments += 1
    
    print(f"Participant {participant_id} complete: {total_segments} total segments saved to {output_folder}")

def main():
    print("=== Gesture Segmentation Script ===")
    print("Step 1: Loading all rest data (p1rest to p6rest)...")
    
    rest_data = load_all_rest_data()
    
    if len(rest_data) == 0:
        print("Error: No rest data found!")
        return
    
    print(f"Loaded {len(rest_data)} rest samples")
    
    print("\nStep 2: Training RestModel...")
    rest_model = RestModel(window_size_ms=20, sampling_rate=200)
    rest_model.train(rest_data)
    
    # Save rest model
    rest_model.save_model('rest_model.pkl')
    
    print("\nStep 3: Segmenting gesture data for all participants...")
    for participant_id in range(1, 7):
        segment_participant_data(participant_id, rest_model)
    
    print("\n=== Segmentation Complete! ===")
    print("New segmented data saved in calibration_data/p{X}new/ folders")

if __name__ == "__main__":
    main()