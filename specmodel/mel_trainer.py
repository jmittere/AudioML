from mel_model import MelTransformerFrameBin, MelTransformerFrame
from mel_dataset import MelMaskedDataset, get_dataset_splits
from audio_train import train_mel, eval_frame, eval_framebin
from mel_utils import write_to_waveform, get_hifi_gan_generator, compare_mels
import argparse
import time

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
    
    parser.add_argument("--model", 
                        type=str, 
                        default="MelTransformerFrame"
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
    MASK_SECONDS= mel_config['mask_seconds'] #in seconds

    #MODEL PARAMS
    D_MODEL = mel_config['d_model']
    N_HEADS = mel_config['n_heads']
    N_LAYERS = mel_config['n_layers']
    D_FF = mel_config['dim_feedforward']
    DROPOUT = mel_config['dropout']
    MAX_SONGS = mel_config['max_songs']

    max_time_frames = int(CLIP_LENGTH * SAMPLE_RATE / HOP_LENGTH)
    
    full_dataset = MelMaskedDataset(
    mel_dir=args.mel_dir,
    mask_seconds=MASK_SECONDS, 
    total_clip_length=CLIP_LENGTH,
    sr=SAMPLE_RATE, 
    hop_length= HOP_LENGTH, 
    max_songs=MAX_SONGS 
    )

    train_set, val_set = get_dataset_splits(full_dataset, args.train_split)
    print("Train dataset length: ", len(train_set))
    print("Validation dataset length: ", len(val_set))

    try:
        if(args.model == "MelTransformerFrameBin"):
            model = MelTransformerFrameBin(
            n_mels=N_MELS,
            max_time_frames=max_time_frames, 
            d_model=D_MODEL, 
            n_heads=N_HEADS, 
            n_layers=N_LAYERS,
            dim_feedforward=D_FF,
            dropout=DROPOUT, 
            )
        elif(args.model == "MelTransformerFrame"):
            model = MelTransformerFrame(
            n_mels=N_MELS, 
            d_model=D_MODEL, 
            n_heads=N_HEADS, 
            n_layers=N_LAYERS,
            dim_feedforward=D_FF,
            dropout=DROPOUT
            )
        else:
            print("Unable to initialize...wrong model name")
            exit()

    except Exception as e: 
        print("Unable to initialize")
        print(e)
        exit()    
    print("--------------------------")
    print(f"Beginning Mel Training for {args.model}...")
    print(f"{args.model}: n_mels: {N_MELS}, d_model: {D_MODEL}, n_heads: {N_HEADS}, n_layers: {N_LAYERS}, dim_feedforward : {D_FF}, dropout: {DROPOUT}")
    print("--------------------------")

    start_time = time.perf_counter()

    train_mel(model_type=args.model, model=model, train_dataset=train_set, val_dataset=val_set, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    seed_frames = int(CLIP_LENGTH - MASK_SECONDS) #in seconds
    if(args.model=="MelTransformerFrame"):
        ground_truth_mel, predicted_mel = eval_frame(model, val_set, seed_seconds=seed_frames)
    elif(args.model=="MelTransformerFrameBin"):
        ground_truth_mel, predicted_mel = eval_framebin(model, val_set, seed_seconds=seed_frames)

    #generator = get_hifi_gan_generator()
    #write_to_waveform("ground_truth_wav.wav", ground_truth_mel, generator, SAMPLE_RATE)
    #write_to_waveform("predicted_wav.wav", predicted_mel, generator, SAMPLE_RATE)
    compare_mels(model_type=args.model, groundtruth=ground_truth_mel, predmel=predicted_mel, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH)


if __name__ == "__main__":
    main()