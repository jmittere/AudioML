from mel_model import MelTransformer
from mel_dataset import MelMaskedDataset, get_dataset_splits
from audio_train import train_mel, eval
from mel_utils import write_to_waveform, get_hifi_gan_generator, compare_mels
import argparse

import json

import sys
sys.path.append('../')

def main(): 
    parser = argparse.ArgumentParser(description="Train model to predict mel spec frames")
    
    parser.add_argument("--epochs", 
                        type=int, 
                        default=10
                        )
    
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=2
                        )
    
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-4
                        )
    
    parser.add_argument("--mask_seconds", 
                        type=float, 
                        default=3.0
                        )
    
    parser.add_argument("--mel_dir", 
                        type=str, 
                        default="../data/mels/1-of-15")
    
    parser.add_argument("--train_split", 
                        type=float, 
                        default=0.8
                        )
    
    parser.add_argument("--seed", 
                        type=int, 
                        default=42
                        )

    args = parser.parse_args()

    try:
        with open('config.json', 'r') as file:
            mel_config = json.load(file)
    except FileNotFoundError:
        print("Error: Mel config.json was not found.")
        exit()

    #MEL SPEC PARAMS
    N_MELS = mel_config['n_mels']
    SAMPLE_RATE = mel_config['sample_rate']
    N_FFT = mel_config['n_fft']
    WIN_LENGTH = mel_config['win_length']
    HOP_LENGTH = mel_config['hop_length']
    FMIN = mel_config['fmin']
    FMAX = mel_config['fmax']
    POWER = mel_config['power']
    LOG_TYPE = mel_config['log_type']
    CLIP_LENGTH= mel_config['clip_length'] #in seconds

    #MODEL PARAMS
    D_MODEL = mel_config['d_model']
    N_HEADS = mel_config['n_heads']
    N_LAYERS = mel_config['n_layers']
    D_FF = mel_config['dim_feedforward']
    DROPOUT = mel_config['dropout']

    full_dataset = MelMaskedDataset(
    mel_dir=args.mel_dir,
    mask_seconds=args.mask_seconds, 
    sr=SAMPLE_RATE, 
    hop_length= HOP_LENGTH
    )

    train_set, val_set = get_dataset_splits(full_dataset, args.train_split)
    print("Train dataset length: ", len(train_set))
    print("Validation dataset length: ", len(val_set))

    try:
        model = MelTransformer(
            n_mels=N_MELS, 
            d_model=D_MODEL, 
            n_heads=N_HEADS, 
            n_layers=N_LAYERS,
            dim_feedforward=D_FF,
            dropout=DROPOUT
        )
    
    except Exception as e: 
        print("Unable to initialize")
        print(e)
        exit()
    
    
    print("Beginning Mel Training")
    train_mel(model=model, train_dataset=train_set, val_dataset=val_set, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    seed_frames = int(CLIP_LENGTH - args.mask_seconds) #in seconds
    ground_truth_mel, predicted_mel = eval(model, val_set, seed_seconds=seed_frames)
    #generator = get_hifi_gan_generator()
    #write_to_waveform("ground_truth_wav.wav", ground_truth_mel, generator, SAMPLE_RATE)
    #write_to_waveform("predicted_wav.wav", predicted_mel, generator, SAMPLE_RATE)
    compare_mels(groundtruth=ground_truth_mel, predmel=predicted_mel, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH)


if __name__ == "__main__":
    main()