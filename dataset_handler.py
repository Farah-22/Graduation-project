# dataset_handler.py
import os
import glob
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def extract_windows(emg, label, window_size, step):
    windows, labels = [], []
    for i in range(0, len(emg) - window_size, step):
        window = emg[i:i + window_size]
        win_label = np.bincount(label[i:i + window_size].flatten()).argmax()
        windows.append(window)
        labels.append(win_label)
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
    return X, Y

class ShiftedEMGDataset(Dataset):
    def __init__(self, original_dataset, shift_factor=0.3, noise_std=0.1):
        self.original_dataset = original_dataset
        self.shift_factor = shift_factor
        self.noise_std = noise_std

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        noise = torch.randn_like(x) * self.noise_std
        shifted_x = x + self.shift_factor + noise
        return shifted_x, y

def apply_scaler(X_raw, scaler):
    b,w,c = X_raw.shape
    return scaler.transform(X_raw.reshape(-1,c)).reshape(b,w,c)

def make_loader(X, Y, batch_size=64, shuffle=True):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y)), batch_size=batch_size, shuffle=shuffle)

def split_base_novel_datasets(full_dataset, base_classes, novel_classes):
    base_indices = []
    novel_indices = []
    
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        label = int(label.item() if torch.is_tensor(label) else label)
        
        if label in base_classes:
            base_indices.append(i)
        elif label in novel_classes:
            novel_indices.append(i)
            
    base_dataset = Subset(full_dataset, base_indices)
    novel_dataset = Subset(full_dataset, novel_indices)
    
    return base_dataset, novel_dataset


TRAIN_DIR = '/kaggle/input/datasets/farahellabban/dataset-ninapro'
TARGET_DIR = '/kaggle/input/datasets/farahellabban/s10-patient'

PRETRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_SUBJECT = [10]


print("Loading Pretrain Data...")
X_pre_train, Y_pre_train = load_ninapro_task(TRAIN_DIR, [0,1,2,3,4,5], PRETRAIN_SUBJECTS, [1,3,4,6])
X_pre_val,   Y_pre_val   = load_ninapro_task(TRAIN_DIR, [0,1,2,3,4,5], PRETRAIN_SUBJECTS, [2,5])
X_pre_train, Y_pre_train = balance_classes(X_pre_train, Y_pre_train)
X_pre_val,   Y_pre_val   = balance_classes(X_pre_val, Y_pre_val)

pre_scaler = StandardScaler()
B,W,C = X_pre_train.shape
X_pre_train_s = pre_scaler.fit_transform(X_pre_train.reshape(-1,C)).reshape(B,W,C)
Bv,Wv,Cv = X_pre_val.shape
X_pre_val_s = pre_scaler.transform(X_pre_val.reshape(-1,Cv)).reshape(Bv,Wv,Cv)

pretrain_loader = DataLoader(TensorDataset(torch.tensor(X_pre_train_s, dtype=torch.float32), torch.tensor(Y_pre_train)), batch_size=64, shuffle=True)
pretrain_val_loader = DataLoader(TensorDataset(torch.tensor(X_pre_val_s, dtype=torch.float32), torch.tensor(Y_pre_val)), batch_size=64, shuffle=False)


print("Loading Subject 10 Data...")
X_t1_train_raw, Y_t1_train = load_ninapro_task(TARGET_DIR, [0,1,2], TARGET_SUBJECT, [1,3,4,6], step=10)
X_t1_test_raw,  Y_t1_test  = load_ninapro_task(TARGET_DIR, [0,1,2], TARGET_SUBJECT, [2,5], step=10)
X_t2_train_raw, Y_t2_train = load_ninapro_task(TARGET_DIR, [0,3,4,5], TARGET_SUBJECT, [1,3,4,6], step=10)
X_t2_test_raw,  Y_t2_test  = load_ninapro_task(TARGET_DIR, [0,3,4,5], TARGET_SUBJECT, [2,5], step=10)

X_t1_train_raw, Y_t1_train = balance_classes(X_t1_train_raw, Y_t1_train)
X_t2_train_raw, Y_t2_train = balance_classes(X_t2_train_raw, Y_t2_train)

subject_scaler = StandardScaler()
B,W,C = X_t1_train_raw.shape
X_t1_train = subject_scaler.fit_transform(X_t1_train_raw.reshape(-1,C)).reshape(B,W,C)

X_t1_test = apply_scaler(X_t1_test_raw, subject_scaler)
X_t2_train = apply_scaler(X_t2_train_raw, subject_scaler)
X_t2_test  = apply_scaler(X_t2_test_raw, subject_scaler)

t1_train_loader = make_loader(X_t1_train, Y_t1_train)
t1_test_loader  = make_loader(X_t1_test, Y_t1_test, shuffle=False)
t2_train_loader = make_loader(X_t2_train, Y_t2_train)
t2_test_loader  = make_loader(X_t2_test, Y_t2_test, shuffle=False)


BASE_CLASSES = [0, 1, 2]
NOVEL_CLASSES = [3, 4, 5]

full_train_dataset = ConcatDataset([t1_train_loader.dataset, t2_train_loader.dataset])
full_test_dataset = ConcatDataset([t1_test_loader.dataset, t2_test_loader.dataset])

base_train_ds, novel_train_ds = split_base_novel_datasets(full_train_dataset, BASE_CLASSES, NOVEL_CLASSES)
base_test_ds, novel_test_ds = split_base_novel_datasets(full_test_dataset, BASE_CLASSES, NOVEL_CLASSES)

BATCH_SIZE = 64
base_train_loader = DataLoader(base_train_ds, batch_size=BATCH_SIZE, shuffle=True)
base_test_loader = DataLoader(base_test_ds, batch_size=BATCH_SIZE, shuffle=False)
novel_train_loader = DataLoader(novel_train_ds, batch_size=BATCH_SIZE, shuffle=False)
novel_test_loader = DataLoader(novel_test_ds, batch_size=BATCH_SIZE, shuffle=False)


target_train_ds = ShiftedEMGDataset(base_train_ds, shift_factor=0.5, noise_std=0.2)
target_test_ds = ShiftedEMGDataset(base_test_ds, shift_factor=0.5, noise_std=0.2)
target_train_loader = DataLoader(target_train_ds, batch_size=64, shuffle=True)
target_test_loader = DataLoader(target_test_ds, batch_size=64, shuffle=False)


if __name__ == '__main__':
    print(" Dataset Handler loaded successfully! All variables are ready to be imported.")