import torch
import numpy as np
import soundfile as sf
import json
import sys
sys.path.append('../hifi-gan')
from models import Generator
from env import AttrDict
import matplotlib.pyplot as plt
import librosa


def write_to_waveform(filename, mel_tensor, generator, sr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_tensor = mel_tensor.T.unsqueeze(0).to(device)

    print("Shape of mel_tensor: " , mel_tensor.size())

    with torch.no_grad():
        audio_hifi = generator(mel_tensor).squeeze().cpu().numpy()

    #audio_hifi /= (np.max(np.abs(audio_hifi)) + 1e-9)

    sf.write(filename, audio_hifi, sr)

def get_hifi_gan_generator():
    # Load config
    with open("../hifi-gan/config_hifigan.json") as f:
        config = json.load(f)

    h = AttrDict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(h).to(device)
    checkpoint = torch.load("../hifi-gan/gen_hifi_gan_02500000",
        map_location=device, 
        weights_only=True
    )

    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def compare_mels(groundtruth, predmel, sample_rate, hop_length):
    #mel specs are (2583, 80) #time x n_mels, librosa expects n_mels x time
    gt_np = groundtruth.detach().cpu().numpy().T
    pred_np = predmel.detach().cpu().numpy().T
    print("gt_np.shape: ", gt_np.shape)
    print("pred_np.shape: ", pred_np.shape)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(gt_np, hop_length=hop_length,  sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title("Ground Truth")
    plt.colorbar(format="%+2.0f", cmap='viridis', label="Log Mel energy (natural log not dB)")


    plt.subplot(1, 2, 2)
    librosa.display.specshow(pred_np, hop_length=hop_length, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title("Prediction")
    plt.colorbar(format="%+2.0f", cmap='viridis', label="Log Mel energy (natural log not dB)")

    plt.savefig("mel_debug.png", dpi=300, bbox_inches="tight")
    plt.close()