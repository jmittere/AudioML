import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import glob
import random

try:
    with open("config.json", "r") as file:
        mel_config = json.load(file)
except FileNotFoundError:
    print("Error: Mel config.json was not found.")
    exit()

SAMPLE_RATE = mel_config["sample_rate"]
N_MELS = mel_config["n_mels"]

def get_dataset_splits(full_dataset, split, seed=42):
    seed_gen = torch.Generator().manual_seed(seed)
    train_size = int(split*len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, val_size], generator=seed_gen)
    return train_dataset, validation_dataset

def collate_fn(batch):
    mels, paths = zip(*batch)
    mels = torch.stack(mels)
    return mels, list(paths)

class MelMaskedDataset(Dataset):
    def __init__(
        self,
        mel_dir,
        n_mels,
        mask_seconds=3.0,
        total_clip_length=10.0,
        sr=22050, #sampling rate of spectrogram
        hop_length=256, #number of samples between frames
        max_songs=527, #number of songs in dataset for training and val
        seed=42, 
        normalize=True,
        stats_path="../mel_stats.npz"
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.mask_seconds = mask_seconds
        self.total_clip_length = total_clip_length
        self.max_songs = max_songs
        self.n_mels = n_mels
        self.normalize = normalize

        self.files = sorted(glob.glob(os.path.join(mel_dir, "**", "*.npy"), recursive=True))

        if self.max_songs is not None:
            random.seed(seed)
            self.files = random.sample(self.files, min(self.max_songs, len(self.files)))

        #number of frames to be masked = mask seconds * sampling rate / hop_length
        self.mask_frames = int(mask_seconds * sr / hop_length)

        if self.normalize:
            stats = np.load(stats_path)
            self.mel_mean = stats["mean"]   #(80,)
            self.mel_std  = stats["std"]    #(80,)
            self.mel_std = np.where(self.mel_std < 1e-6, 1e-6, self.mel_std)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        mel = np.load(filepath)      # (80, T)
        mel = mel.T                  # (T, 80)

        TARGET_FRAMES = int(self.total_clip_length * self.sr / self.hop_length)

        if mel.shape[0] > TARGET_FRAMES:
            mel = mel[-TARGET_FRAMES:]
        elif mel.shape[0] < TARGET_FRAMES:
            pad_len = TARGET_FRAMES - mel.shape[0]

            #pad with mean instead of 0 after norm
            if self.normalize:
                pad = np.tile(self.mel_mean, (pad_len, 1))
            else:
                pad = np.zeros((pad_len, self.n_mels))

            mel = np.concatenate([mel, pad], axis=0)

        if self.normalize:
            mel = (mel - self.mel_mean) / self.mel_std

        return torch.from_numpy(mel).float(), filepath  # (T, 80)
    
    def get_mel_shape(self, idx=1):
        filepath = self.files[idx]
        mel = np.load(filepath)
        mel = mel.T
        return mel.shape, type(mel)
    