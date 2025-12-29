import numpy as np
# Model 1 Parameters
WINDOW_SIZE = 20
THRESHOLD_FACTOR = 6.0  # Artırıldı: Gürültüyü hareket sanmaması için
MIN_DURATION = 20
PADDING = 0        
class RestDetector:
    """
    Model 1: Energy-based segmentation model.
    Detects active gestures by comparing EMG energy against a learned noise threshold.
    """
    def __init__(self, window_size=20, threshold_factor=3.0, min_duration=20, padding=0):
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.min_duration = min_duration
        self.padding = padding
        self.threshold = None

    def fit(self, rest_data_list):
        """
        Calibrates the threshold using a list of rest data arrays.
        Threshold = Mean_Noise + (Factor * Std_Noise)
        """
        energies = []
        for data in rest_data_list:
            processed = self._preprocess(data)
            energy = self._compute_energy(processed)
            energies.append(energy)
        
        if not energies:
            print("Warning: No rest data provided for calibration.")
            return False
            
        # Concatenate all rest energies to find global statistics
        full_energy = np.concatenate(energies)
        mean_noise = np.mean(full_energy)
        std_noise = np.std(full_energy)
        
        self.threshold = mean_noise + (self.threshold_factor * std_noise)
        
        print(f"--- Model 1 Calibration ---")
        print(f"Mean Noise Energy: {mean_noise:.4f}")
        print(f"Std Dev Noise    : {std_noise:.4f}")
        print(f"Calculated Threshold: {self.threshold:.4f}")
        return True

    def predict(self, window_data):
        """
        Gerçek zamanlı (Real-time) kullanım için tahmin fonksiyonu.
        Verilen pencere (örn: 20 sample) REST ise True, ACTIVE ise False döner.
        """
        if self.threshold is None:
            return True # Eğitilmediyse varsayılan olarak rest kabul et

        processed = self._preprocess(window_data)
        
        # EMG kolonlarını seç
        n_cols = processed.shape[1]
        emg_cols = self._get_emg_indices(n_cols)
        emg_data = processed[:, emg_cols] if emg_cols else processed
        
        # Bu pencerenin ortalama enerjisini hesapla
        energy = np.mean(np.abs(emg_data))
        
        # Enerji eşikten düşükse -> REST (True), yüksekse -> ACTIVE (False)
        return energy < self.threshold

    def segment(self, data):
        """
        Segments the input data into active gesture clips.
        Returns a list of numpy arrays (one for each detected gesture).
        """
        if self.threshold is None:
            raise RuntimeError("RestDetector is not fitted. Call fit() with rest data first.")

        processed = self._preprocess(data)
        energy = self._compute_energy(processed)
        
        # Create a boolean mask where energy is above threshold
        active_mask = energy > self.threshold
        
        # Find contiguous segments
        segments_indices = self._find_contiguous_segments(active_mask)
        
        cropped_segments = []
        for start, end in segments_indices:
            # Apply padding (add context before/after)
            s = max(0, start - self.padding)
            e = min(len(data), end + self.padding)
            
            # Check if segment is long enough
            if (e - s) >= self.min_duration:
                cropped_segments.append(data[s:e])
                
        return cropped_segments

    def _preprocess(self, data):
        """
        Removes DC offset (bias) from EMG channels.
        Assumes 34 columns: Myo1 (0-16), Myo2 (17-33).
        EMG indices: 0-7 and 17-24.
        """
        processed_data = data.copy()
        n_cols = data.shape[1]
        emg_cols = self._get_emg_indices(n_cols)
            
        if emg_cols:
            # Center data around 0 (remove bias)
            mean_vals = np.mean(processed_data[:, emg_cols], axis=0)
            processed_data[:, emg_cols] -= mean_vals
            
        return processed_data

    def _compute_energy(self, data):
        """
        Computes sliding window energy (Mean Absolute Value).
        """
        n_cols = data.shape[1]
        emg_cols = self._get_emg_indices(n_cols)
        
        # Use only EMG data for energy calculation
        emg_data = data[:, emg_cols] if emg_cols else data
        
        abs_data = np.abs(emg_data)
        # Average energy across all EMG channels
        mean_energy = np.mean(abs_data, axis=1)
        
        # Sliding window smoothing (Convolution)
        window = np.ones(self.window_size) / self.window_size
        return np.convolve(mean_energy, window, mode='same')

    def _get_emg_indices(self, n_cols):
        emg_cols = []
        if n_cols >= 8: emg_cols.extend(range(0, 8))
        if n_cols >= 25: emg_cols.extend(range(17, 25))
        return emg_cols

    def _find_contiguous_segments(self, mask):
        """
        Helper to find start and end indices of True regions.
        """
        # Pad with False to detect edges at start/end
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        merged = []
        if len(starts) > 0:
            curr_start, curr_end = starts[0], ends[0]
            for i in range(1, len(starts)):
                # Merge segments that are very close (less than window size gap)
                if (starts[i] - curr_end) < self.window_size:
                    curr_end = ends[i]
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = starts[i], ends[i]
            merged.append((curr_start, curr_end))
            
        return merged