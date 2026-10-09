# dataset_handler.py
import os
import glob
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, iirnotch

def apply_myo_filters(emg_data, fs=200.0):
    w0 = 50.0 / (fs / 2.0)
    b_notch, a_notch = iirnotch(w0, Q=30.0)
    emg_filtered = lfilter(b_notch, a_notch, emg_data, axis=0)
    
    low, high = 20.0 / (fs / 2.0), 90.0 / (fs / 2.0)
    b_band, a_band = butter(4, [low, high], btype='band')
    return lfilter(b_band, a_band, emg_filtered, axis=0)

def extract_windows(emg, label, window_size, step):
    windows, labels = [], []
    for i in range(0, len(emg) - window_size, step):
        windows.append(emg[i:i + window_size])
        labels.append(np.bincount(label[i:i + window_size].flatten()).argmax())
    return np.array(windows), np.array(labels)

def load_ninapro_task(base_path, allowed_classes, subjects, allowed_reps, window_size=200, step=20, num_channels=8):
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
                
                emg = apply_myo_filters(emg, fs=200.0)
                
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

def balance_classes(X, Y):
    classes, counts = np.unique(Y, return_counts=True)
    print(f"  Before Balancing: {dict(zip(classes, counts))}")
    rest_idx = np.where(Y == 0)[0]
    move_idx = np.where(Y != 0)[0]
    if len(move_idx) > 0 and len(rest_idx) > 0:
        target_rest = int(np.mean([c for cls, c in zip(classes, counts) if cls != 0]))
        if len(rest_idx) > target_rest:
            np.random.seed(42)
            sel_rest = np.random.choice(rest_idx, target_rest, replace=False)
            final_idx = np.concatenate([sel_rest, move_idx])
            np.random.shuffle(final_idx)
            X, Y = X[final_idx], Y[final_idx]
            nc, ncounts = np.unique(Y, return_counts=True)
            print(f"  After Balancing: {dict(zip(nc, ncounts))}")
    return X, Y

def make_loader(X, Y, batch_size=64, shuffle=True):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle
    )