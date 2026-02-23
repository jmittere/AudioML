from mel_model import MelTransformer
from mel_dataset import MelMaskedDataset
from mel_audio_train import train_mel, eval
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
    
    parser.add_argument("--mel_dir", 
                        type=str, 
                        default="../data/mels/0-of-15")

    args = parser.parse_args()

    dataset = MelMaskedDataset(
    mel_dir="../data/mels/0-of-15",
    mask_seconds=10.0
    )

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
    train_mel(model=model, dataset=dataset, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()