# segment_gestures.py
import numpy as np
import glob
import os
from rest_model import RestModel

def segment_gesture_file(gesture_data, rest_model, min_segment_length_ms=100):
    """
    Segment a single gesture file by removing rest periods
    
    gesture_data: numpy array (time_steps, 34)
    rest_model: trained RestModel
    min_segment_length_ms: minimum length for valid segment
    
    Returns: list of segmented numpy arrays
    """
    min_samples = int((min_segment_length_ms / 1000) * rest_model.sampling_rate)
    
    print(f"\n    [DEBUG] Input data shape: {gesture_data.shape}")
    print(f"    [DEBUG] Window size: {rest_model.samples_per_window} samples")
    print(f"    [DEBUG] Stride: {rest_model.stride} samples")
    print(f"    [DEBUG] Min segment length: {min_samples} samples ({min_segment_length_ms}ms)")
    
    # Slide 20ms window through data with 50% overlap
    num_windows = (len(gesture_data) - rest_model.samples_per_window) // rest_model.stride + 1
    print(f"    [DEBUG] Number of windows to process: {num_windows}")
    
    # Predict rest/non-rest for each window
    predictions = []
    rest_count = 0
    non_rest_count = 0
    
    for i in range(num_windows):
        start_idx = i * rest_model.stride
        end_idx = start_idx + rest_model.samples_per_window
        window = gesture_data[start_idx:end_idx]
        
        features = rest_model.extract_features(window)
        is_rest = rest_model.is_rest(features)
        predictions.append((start_idx, end_idx, is_rest))
        
        if is_rest:
            rest_count += 1
        else:
            non_rest_count += 1
    
    print(f"    [DEBUG] Predictions - Rest: {rest_count}, Non-rest: {non_rest_count}")
    
    # Print first 10 and last 10 predictions to see pattern
    print(f"    [DEBUG] First 10 predictions: {['R' if p[2] else 'G' for p in predictions[:10]]}")
    print(f"    [DEBUG] Last 10 predictions: {['R' if p[2] else 'G' for p in predictions[-10:]]}")
    
    # Find continuous non-rest segments
    segments = []
    current_segment_start = None
    segment_id = 0
    
    for idx, (start_idx, end_idx, is_rest) in enumerate(predictions):
        if not is_rest:  # Non-rest window
            if current_segment_start is None:
                current_segment_start = start_idx
                print(f"    [DEBUG] Segment {segment_id} START at window {idx} (sample {start_idx})")
        else:  # Rest window
            if current_segment_start is not None:
                # End of non-rest segment
                segment_data = gesture_data[current_segment_start:end_idx]
                segment_length = len(segment_data)
                
                print(f"    [DEBUG] Segment {segment_id} END at window {idx} (sample {end_idx})")
                print(f"    [DEBUG]   -> Length: {segment_length} samples ({segment_length/rest_model.sampling_rate*1000:.1f}ms)")
                
                # Only keep if long enough
                if segment_length >= min_samples:
                    segments.append(segment_data)
                    print(f"    [DEBUG]   -> ✓ KEPT (>= {min_samples} samples)")
                else:
                    print(f"    [DEBUG]   -> ✗ DISCARDED (< {min_samples} samples)")
                
                current_segment_start = None
                segment_id += 1
    
    # Handle case where file ends with non-rest
    if current_segment_start is not None:
        segment_data = gesture_data[current_segment_start:]
        segment_length = len(segment_data)
        
        print(f"    [DEBUG] Segment {segment_id} END at file end (sample {len(gesture_data)})")
        print(f"    [DEBUG]   -> Length: {segment_length} samples ({segment_length/rest_model.sampling_rate*1000:.1f}ms)")
        
        if segment_length >= min_samples:
            segments.append(segment_data)
            print(f"    [DEBUG]   -> ✓ KEPT (>= {min_samples} samples)")
        else:
            print(f"    [DEBUG]   -> ✗ DISCARDED (< {min_samples} samples)")
    
    print(f"    [DEBUG] Total segments found: {len(segments)}")
    
    return segments
def segment_participant_data(participant_id, rest_model):
    """
    Process all gesture files for one participant
    
    participant_id: 1-6
    rest_model: trained RestModel (participant-specific)
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
    print("=== Gesture Segmentation Script (Participant-Specific) ===")
    print("This will segment each participant's data using their own rest data")
    print()
    
    for participant_id in range(1, 7):
        print(f"\n{'='*60}")
        print(f"PARTICIPANT {participant_id}")
        print(f"{'='*60}")
        
        # Load THIS participant's rest data only
        rest_folder = f'calibration_data/p{participant_id}rest'
        if not os.path.exists(rest_folder):
            print(f"❌ Error: {rest_folder} not found! Skipping participant {participant_id}")
            continue
        
        rest_data = []
        files = glob.glob(f'{rest_folder}/*.npy')
        
        if len(files) == 0:
            print(f"❌ No rest data files found in {rest_folder}, skipping...")
            continue
        
        print(f"Loading rest data from {rest_folder}...")
        for file in files:
            data = np.load(file)
            rest_data.append(data)
            print(f"  ✓ Loaded: {os.path.basename(file)} (shape: {data.shape})")
        
        print(f"\nTraining RestModel for participant {participant_id} on {len(rest_data)} rest samples...")
        rest_model = RestModel(window_size_ms=20, sampling_rate=200)
        rest_model.train(rest_data)
        
        # Save participant-specific rest model
        rest_model.save_model(f'rest_model_p{participant_id}.pkl')
        print(f"✓ Saved rest_model_p{participant_id}.pkl")
        
        # Segment THIS participant's gesture data
        segment_participant_data(participant_id, rest_model)
    
    print("\n" + "="*60)
    print("=== Segmentation Complete! ===")
    print("="*60)
    print("New segmented data saved in calibration_data/p{X}new/ folders")
    print("Participant-specific rest models saved as rest_model_p{X}.pkl")

if __name__ == "__main__":
    main()