import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

def get_dataset_splits(full_dataset, split, seed=None):
    #seed_gen = torch.Generator().manual_seed(seed)
    train_size = int(split*len(full_dataset))
    val_size = len(full_dataset) - train_size
    #train_dataset, validation_dataset = random_split(full_dataset, [train_size, val_size], generator=seed_gen)
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, validation_dataset

def collate_fn(batch):
    mel_inputs, mel_targets, mask_starts, _ = zip(*batch)

    mel_inputs = torch.stack(mel_inputs)      # (B, T, 80)
    mel_targets = torch.stack(mel_targets)    # (B, mask_T, 80)
    mask_starts = torch.tensor(mask_starts)

    return mel_inputs, mel_targets, mask_starts

class MelMaskedDataset(Dataset):
    def __init__(
        self,
        mel_dir,
        mask_seconds=10.0,
        sr=22050, #sampling rate of spectrogram
        hop_length=256 #number of samples between frames
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.mask_seconds = mask_seconds
        
        self.files = []
        for filename in sorted(os.listdir(mel_dir)):
            if filename.endswith(".npy"):
                full_path = os.path.join(mel_dir, filename)
                self.files.append(full_path)

        #number of frames to be masked = mask seconds * sampling rate / hop_length
        self.mask_frames = int(mask_seconds * sr / hop_length)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        mel = np.load(filepath)      # (80, T)
        mel = mel.T  # (T, 80)

        TARGET_FRAMES = int(30 * self.sr / self.hop_length) #expected num of frames in each 30 second snippet
        if mel.shape[0] > TARGET_FRAMES: #if greater than 30s slighty, cut off ending frames
            mel = mel[:TARGET_FRAMES]
        elif mel.shape[0] < TARGET_FRAMES: #if less, padd zeros at the end
            pad_len = TARGET_FRAMES - mel.shape[0]
            pad = np.zeros((pad_len, 80))
            mel = np.concatenate([mel, pad], axis=0)

        T = mel.shape[0]
        #starting Time index of mask
        mask_start = T - self.mask_frames

        # Input: mask the tail
        mel_input = mel.copy()
        mel_input[mask_start:] = 0.0

        # Target: only the masked region
        mel_target = mel[mask_start:]

        return (
            torch.from_numpy(mel_input).float(),   # (T, 80)
            torch.from_numpy(mel_target).float(),  # (mask_T, 80)
            mask_start, 
            filepath
        )