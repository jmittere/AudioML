from mel_model import MelTransformer
from mel_dataset import MelMaskedDataset, get_dataset_splits
from audio_train import train_mel, eval
from mel_utils import write_to_waveform, get_hifi_gan_generator, compare_mels
import argparse

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
                        default=5.0
                        )
    
    parser.add_argument("--mel_dir", 
                        type=str, 
                        default="../data/mels/0-of-15")
    
    parser.add_argument("--train_split", 
                        type=float, 
                        default=0.8
                        )
    
    parser.add_argument("--seed", 
                        type=int, 
                        default=42
                        )

    args = parser.parse_args()

    SAMPLE_RATE = 22050

    full_dataset = MelMaskedDataset(
    mel_dir=args.mel_dir,
    mask_seconds=args.mask_seconds
    )
    train_set, val_set = get_dataset_splits(full_dataset, args.train_split)
    print("Train dataset length: ", len(train_set))
    print("Validation dataset length: ", len(val_set))

    try:
        model = MelTransformer(
            n_mels=80, 
            d_model=128, 
            n_heads=4, 
            n_layers=4,
            dropout=0.1
        )
    
    except Exception as e: 
        print("Unable to initialize")
        print(e)
        exit()
    
    
    print("Beginning Mel Training")
    train_mel(model=model, train_dataset=train_set, val_dataset=val_set, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    ground_truth_mel, predicted_mel = eval(model, val_set)
    generator = get_hifi_gan_generator()
    write_to_waveform("ground_truth_wav.wav", ground_truth_mel, generator, SAMPLE_RATE)
    write_to_waveform("predicted_wav.wav", predicted_mel, generator, SAMPLE_RATE)
    compare_mels(groundtruth=ground_truth_mel, predmel=predicted_mel, sample_rate=SAMPLE_RATE)


if __name__ == "__main__":
    main()