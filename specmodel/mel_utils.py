import torch
import numpy as np
import soundfile as sf
import json
import sys
sys.path.append("../hifi-gan")
from models import Generator
from env import AttrDict
import matplotlib.pyplot as plt
import librosa
import os
from datetime import datetime

stats = np.load("../mel_stats.npz")
mel_mean = stats["mean"]
mel_std  = stats["std"]

def write_to_waveform(filename, mel_tensor, generator, sr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_tensor = mel_tensor.T.unsqueeze(0).to(device)

    #must be (80, T) for HIFIGAN
    #print("Shape of mel_tensor: " , mel_tensor.size())

    with torch.no_grad():
        audio_hifi = generator(mel_tensor).squeeze().cpu().numpy()

    #audio_hifi /= (np.max(np.abs(audio_hifi)) + 1e-9)

    sf.write(filename, audio_hifi, sr)

def get_hifi_gan_generator():
    # Load config
    with open("../hifi-gan/config_hifigan.json", encoding="utf-8") as f:
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

def compare_mels(filepath, model_type, groundtruth, predmel, baseline, sample_rate, hop_length, output_dir="../outputs"):
    #mel specs are (2583, 80) #time x n_mels, librosa expects n_mels x time
    gt_np = groundtruth.detach().cpu().numpy().T
    pred_np = predmel.detach().cpu().numpy().T
    baseline_np = baseline.detach().cpu().numpy().T

    gt_np = gt_np * mel_std[:, None] + mel_mean[:, None]
    pred_np = pred_np * mel_std[:, None] + mel_mean[:, None]
    baseline_np = baseline_np * mel_std[:, None] + mel_mean[:, None]

    #print("gt_np.shape: ", gt_np.shape)
    #print("pred_np.shape: ", pred_np.shape)

    #shared color scale
    vmin = min(gt_np.min(), pred_np.min(), baseline_np.min())
    vmax = max(gt_np.max(), pred_np.max(), baseline_np.max())

    plt.figure(figsize=(18, 5))

    filename = os.path.basename(filepath).rstrip(".npy")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plt.suptitle(f"{model_type} | {filepath} | {timestamp}", fontsize=12)

    # --- Ground Truth ---
    plt.subplot(1, 3, 1)
    img1 = librosa.display.specshow(
        gt_np,
        hop_length=hop_length,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        cmap="plasma",
        vmin=vmin,
        vmax=vmax
    )
    plt.title("Ground Truth")
    plt.colorbar(img1, format="%+2.0f", label="Log Mel energy (natural log not dB)")

    # --- Prediction ---
    plt.subplot(1, 3, 2)
    img2 = librosa.display.specshow(
        pred_np,
        hop_length=hop_length,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        cmap="plasma",
        vmin=vmin,
        vmax=vmax
    )
    plt.title("Prediction")
    plt.colorbar(img2, format="%+2.0f", label="Log Mel energy (natural log not dB)")

    # --- Baseline ---
    plt.subplot(1, 3, 3)
    img3 = librosa.display.specshow(
        baseline_np,
        hop_length=hop_length,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        cmap="plasma",
        vmin=vmin,
        vmax=vmax
    )
    plt.title("Baseline")
    plt.colorbar(img3, format="%+2.0f", label="Log Mel energy (natural log not dB)")
    try:
        plt.savefig(f"{output_dir}/mel_debug_{model_type}_{filename}.png", dpi=300, bbox_inches="tight")
    except OSError as e:
        print(e) 
        print(f"Unable to savefig for {filename}")
    plt.close()

def generate_waveforms(filepath, model_type, groundtruth, predmel, baseline, sample_rate, n_fft, hop_length, n_iter, win_length, fmin, fmax, power, output_dir="../outputs"):
    #generate waveforms of predicted and also ground truth with griffin lim for a fair reconstruction comparison
    gt_np = groundtruth.detach().cpu().numpy().T
    pred_np = predmel.detach().cpu().numpy().T
    baseline_np = baseline.detach().cpu().numpy().T

    #print("gt_np.shape: ", gt_np.shape)
    #print("pred_np.shape: ", pred_np.shape)
    
    #denorm and then undo log
    gt_np = gt_np * mel_std[:, None] + mel_mean[:, None]
    pred_np = pred_np * mel_std[:, None] + mel_mean[:, None]
    baseline_np = baseline_np * mel_std[:, None] + mel_mean[:, None]

    # Undo natural log
    gt_np = np.exp(gt_np)
    pred_np = np.exp(pred_np)
    baseline_np = np.exp(baseline_np)

    filename = os.path.basename(filepath).rstrip(".npy")
    
    audio_pred = _get_waveform(pred_np, sample_rate, n_fft, hop_length, n_iter, win_length, fmin, fmax, power)
    audio_gt = _get_waveform(gt_np, sample_rate, n_fft, hop_length, n_iter, win_length, fmin, fmax, power)
    #audio_base = _get_waveform(baseline_np, sample_rate, n_fft, hop_length, n_iter, win_length, fmin, fmax, power)

    sf.write(f"{output_dir}/mel_debug_{model_type}_{filename}_pred.wav", audio_pred, sample_rate)
    sf.write(f"{output_dir}/mel_debug_{model_type}_{filename}_gt.wav", audio_gt, sample_rate)
    #sf.write(f"{output_dir}/mel_debug_{model_type}_{filename}_bl.wav", audio_base, sample_rate)


def _get_waveform(mel, sample_rate, n_fft, hop_length, n_iter, win_length, fmin, fmax, power):
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter, #number of iterations for griffin lim
        win_length=win_length, 
        #fmin=fmin, 
        #fmax=fmax, 
        power=power
    )
    #ensure amplitudes are in safe range and doesn't blow speakers
    audio /= (np.max(np.abs(audio)) + 1e-9)
    return audio