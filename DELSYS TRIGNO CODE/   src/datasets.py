import torch
from torch.utils.data import Dataset

class ShiftedEMGDataset(Dataset):
    """
    Simulates real-world Domain Shift (e.g., electrode displacement, sweat)
    by adding statistical noise and feature offsets to the original HD-sEMG data.
    Used for Unsupervised Domain Adaptation (UDA) stress testing.
    """
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