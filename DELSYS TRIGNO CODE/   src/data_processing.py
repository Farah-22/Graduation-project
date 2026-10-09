import os
import glob
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, iirnotch, resample

# ==========================================
# 1. Signal Processing & Filtering
# ==========================================

def apply_emg_filters(emg_data, fs=2000.0):
    """
    Applies Notch and Bandpass filters to raw EMG data.
    
    Args:
        emg_data: Raw EMG signal (Time, Channels).
        fs: Sampling frequency (2000.0 Hz for NinaPro DB2).
        
    Returns:
        Filtered EMG signal.
    """
    # 1. Notch Filter (50 Hz) to remove power line interference
    w0 = 50.0 / (fs / 2.0)
    Q = 30.0
    b_notch, a_notch = iirnotch(w0, Q)
    emg_filtered = lfilter(b_notch, a_notch, emg_data, axis=0)
    
    # 2. Bandpass Filter (20-500 Hz) to remove motion artifacts and high-frequency noise
    low = 20.0 / (fs / 2.0)
    high = 500.0 / (fs / 2.0)
    b_band, a_band = butter(4, [low, high], btype='band')
    emg_filtered = lfilter(b_band, a_band, emg_filtered, axis=0)
    
    return emg_filtered

def apply_lowpass_envelope(emg_data, fs=2000.0, cutoff=10.0):
    """
    Applies a low-pass filter to smooth the signal after taking the absolute value (Envelope).
    
    Args:
        emg_data: Absolute EMG signal.
        fs: Sampling frequency.
        cutoff: Cutoff frequency for the envelope (10.0 Hz).
        
    Returns:
        Smoothed signal envelope.
    """
    normal_cutoff = cutoff / (fs / 2.0)
    b_low, a_low = butter(4, normal_cutoff, btype='low', analog=False)
    envelope = lfilter(b_low, a_low, emg_data, axis=0)
    return envelope

# ==========================================
# 2. Windowing
# ==========================================

def extract_windows(emg, label, window_size, step):
    """
    Extracts overlapping windows from continuous EMG data.
    
    Args:
        emg: Processed EMG signal.
        label: Corresponding labels.
        window_size: Number of samples per window.
        step: Step size for sliding window.
        
    Returns:
        Tuple of (windows, labels).
    """
    windows, labels = [], []
    for i in range(0, len(emg) - window_size, step):
        window = emg[i:i + window_size]
        # Assign the most frequent label in the window
        win_label = np.bincount(label[i:i + window_size].flatten()).argmax()
        windows.append(window)
        labels.append(win_label)
    return np.array(windows), np.array(labels)

# ==========================================
# 3. Data Loading (NinaPro DB2)
# ==========================================

def load_ninapro_task(base_path, allowed_classes, subjects, allowed_reps, 
                      train_reps=[1, 3, 4, 6], window_size=400, step=40, 
                      num_channels=12, mu=255.0):
    """
    Loads, processes, and segments NinaPro DB2 EMG data.
    
    Args:
        base_path: Root directory of the dataset.
        allowed_classes: List of movement classes to include.
        subjects: List of subject IDs to load.
        allowed_reps: List of repetitions to include.
        train_reps: Repetitions used for fitting the StandardScaler.
        window_size: Window length in samples.
        step: Sliding window step.
        num_channels: Number of EMG channels to use.
        mu: Mu-law compression parameter.
        
    Returns:
        Tuple of (X, Y) arrays.
    """
    all_x, all_y = [], []
    
    for subj in subjects:
        search_pattern = os.path.join(base_path, '**', f'S{subj}_E2*.mat')
        file_paths = glob.glob(search_pattern, recursive=True)
        
        for file_path in file_paths:
            try:
                data = sio.loadmat(file_path)
                emg = data['emg'][:, :num_channels]
                labels = data['restimulus'].flatten()
                reps = data['repetition'].flatten()

                # 1. Apply basic filters first
                emg = apply_emg_filters(emg, fs=2000.0)

                # 2. Mathematical operations to resolve crosstalk (Absolute value)
                emg = np.abs(emg)
                
                # 3. Extract smooth signal envelope
                emg = apply_lowpass_envelope(emg, fs=2000.0, cutoff=10.0)
                
                # 4. Clip negative values caused by filter then apply logarithm (Mu-law)
                emg = np.clip(emg, a_min=0, a_max=None)
                emg = np.log1p(mu * emg) / np.log1p(mu)
                
                # 5. Standard Scaling
                scaler = StandardScaler()
                train_mask = np.isin(reps, train_reps)
                if np.any(train_mask):
                    scaler.fit(emg[train_mask])
                    emg = scaler.transform(emg)
                else:
                    emg = scaler.fit_transform(emg)

                # 6. Segment extraction and Transition Cropping
                unique_reps = np.unique(reps)
                for r in unique_reps:
                    if r not in allowed_reps:
                        continue
                        
                    for c in allowed_classes:
                        segment_mask = (labels == c) & (reps == r)
                        if not np.any(segment_mask):
                            continue
                            
                        segment_emg = emg[segment_mask]
                        segment_label = labels[segment_mask]
                        
                        # Transition Cropping: Crop 15% from start and end for non-rest classes
                        if c != 0: 
                            margin = int(len(segment_emg) * 0.15)
                            if len(segment_emg) > (2 * margin + window_size):
                                segment_emg = segment_emg[margin:-margin]
                                segment_label = segment_label[margin:-margin]
                                
                        if len(segment_emg) >= window_size:
                            x_wins, y_wins = extract_windows(segment_emg, segment_label, window_size, step)
                            all_x.append(x_wins)
                            all_y.append(y_wins)
                            
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if not all_x:
        return np.array([]), np.array([])

    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0).astype(np.int64)
    return X, Y

# ==========================================
# 4. Data Balancing & DataLoader Creation
# ==========================================

def balance_classes(X, Y):
    """
    Balances the dataset by downsampling the majority class (Rest/0).
    
    Args:
        X: Feature array.
        Y: Label array.
        
    Returns:
        Balanced (X, Y) arrays.
    """
    classes, counts = np.unique(Y, return_counts=True)
    print(f"  Before Balancing : {dict(zip(classes, counts))}")
    
    rest_idx = np.where(Y == 0)[0]
    move_idx = np.where(Y != 0)[0]
    
    if len(move_idx) > 0 and len(rest_idx) > 0:
        # Target rest count is the average of movement class counts
        target_rest = int(np.mean([c for cls, c in zip(classes, counts) if cls != 0]))
        
        if len(rest_idx) > target_rest:
            np.random.seed(42)
            sel_rest = np.random.choice(rest_idx, target_rest, replace=False)
            final_idx = np.concatenate([sel_rest, move_idx])
            np.random.shuffle(final_idx)
            
            X, Y = X[final_idx], Y[final_idx]
            
            nc, ncounts = np.unique(Y, return_counts=True)
            print(f"  After Balancing  : {dict(zip(nc, ncounts))}")
            
    return X, Y

def make_loader(X, Y, batch_size=64, shuffle=True):
    """
    Creates a PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Feature array.
        Y: Label array.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        
    Returns:
        PyTorch DataLoader.
    """
    # Data is already clean and normalized, send directly to Tensor
    # Note: Labels are cast to torch.long for compatibility with loss functions
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), 
        torch.tensor(Y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ==========================================
# 5. Universal Device Adapter (New Addition)
# ==========================================

class UniversalEMGPreprocessor:
    """
    Converts EMG data from any device (varying channels and sampling rates) 
    into the unified format expected by the model (default: 12 channels, 2000Hz).
    """
    def __init__(self, target_channels=12, target_fs=2000.0):
        self.target_channels = target_channels
        self.target_fs = target_fs
        
    def adapt_to_device(self, raw_emg, source_fs, source_channels):
        """
        Adapts raw EMG signal to the target format.
        
        Args:
            raw_emg: Raw signal array (Time, Channels).
            source_fs: Original sampling frequency.
            source_channels: Original number of channels.
            
        Returns:
            Normalized and resampled EMG signal.
        """
        # Step 1: Resampling (Time alignment)
        if source_fs != self.target_fs:
            num_samples = int(raw_emg.shape[0] * self.target_fs / source_fs)
            resampled_emg = np.zeros((num_samples, source_channels))
            for ch in range(source_channels):
                resampled_emg[:, ch] = resample(raw_emg[:, ch], num_samples)
        else:
            resampled_emg = raw_emg
            
        # Step 2: Channel Mapping (Spatial alignment)
        if source_channels < self.target_channels:
            # If fewer channels, repeat available channels to fill the gap
            normalized_emg = np.zeros((resampled_emg.shape[0], self.target_channels))
            for i in range(self.target_channels):
                normalized_emg[:, i] = resampled_emg[:, i % source_channels]
        elif source_channels > self.target_channels:
            # If more channels, take only the first N channels
            normalized_emg = resampled_emg[:, :self.target_channels]
        else:
            normalized_emg = resampled_emg
            
        # Step 3: Statistical Normalization (Z-score)
        mean = np.mean(normalized_emg, axis=0, keepdims=True)
        std = np.std(normalized_emg, axis=0, keepdims=True) + 1e-8
        normalized_emg = (normalized_emg - mean) / std
        
        return normalized_emg