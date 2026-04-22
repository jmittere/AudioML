#Preprocessing script to convert waveforms to log-mel magnitude spectrograms
#compressing with natural log for compatibility with HIFI-GAN vocoder
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import soundfile as sf
import pandas as pd
import re
import json
import glob
from tqdm import tqdm

try:
    with open('./specmodel/config.json', 'r') as file:
        mel_config = json.load(file)
except FileNotFoundError:
    print("Error: Mel config.json was not found.")
    exit()

N_MELS = mel_config['n_mels']
SAMPLE_RATE = mel_config['sample_rate']
N_FFT = mel_config['n_fft']
WIN_LENGTH = mel_config['win_length']
HOP_LENGTH = mel_config['hop_length']
FMIN = mel_config['fmin']
FMAX = mel_config['fmax']
POWER = mel_config['power']
LOG_TYPE = mel_config['log_type']
CLIP_LENGTH = mel_config['clip_length'] #clips the length of mel spec to this size in seconds

def decode_row(audio_dict):
    y, sr = librosa.load(io.BytesIO(audio_dict['bytes']),
                         sr=None, #keep same sample rate
                         mono=True) #convert to mono if stereo
    return y, sr

def sanitize(text):
    text = text.replace(" ", "")
    text = re.sub(r'[\\/*?:"<>|]', "", text)  #remove illegal chars
    return text

def convert_parquet_to_waveform():
        
    counter = 0
    for i in range(0,15):
        
        #load all parquet files and convert them to their raw waveforms
        out_dir = f"./data/waveforms/{i}-of-15"
        os.makedirs(out_dir, exist_ok=True)

        if(i<10):
            df = pd.read_parquet(f'./data/raw_parquets/train-0000{str(i)}-of-00015.parquet', engine="pyarrow")
        else:
            df = pd.read_parquet(f'./data/raw_parquets/train-000{str(i)}-of-00015.parquet', engine="pyarrow")

        for idx, row in df.iterrows():
                    try:
                        y, sr = decode_row(row['audio'])
                        title = sanitize(row['title'])
                        artist = sanitize(row['artist'])              
                        sf.write(f"{out_dir}/{idx}_{title}_{artist}.wav", y, sr, subtype="PCM_16")
                        counter+=1
                    except Exception as e:
                        print(e)
                        print(f"{row['title']}, {row['artist']}, {row['language']}") 

    print("Number of Files converted to waveform: ", counter)

def convert_waveform_to_mel_spec():
    #Load the audio files from waveforms folder and convert to mel spectrograms
    total_counter = 0

    max_len = int(SAMPLE_RATE * CLIP_LENGTH)

    for i in range(0,15):
        counter = 0
        directory = f'./data/waveforms/{i}-of-15'
        out_dir = f'./data/mels/{i}-of-15'
        os.makedirs(out_dir, exist_ok=True)


        for entry in os.scandir(directory):  
            if entry.is_file():
                filename = entry.name.split(".wav")[0]
                samples, sample_rate = librosa.load(entry.path, sr=SAMPLE_RATE, mono=True) #consistent sampling rate and mono audio for all samples

                #trim clips to 10 seconds
                if len(samples) > max_len:
                    samples = samples[:max_len]

                #params needed for HIFI-GAN vocoder for output post processing
                mel = librosa.feature.melspectrogram(y=samples,
                                                    sr=sample_rate,
                                                    n_fft=N_FFT,
                                                    hop_length=HOP_LENGTH,
                                                    win_length=WIN_LENGTH,
                                                    fmin=FMIN, 
                                                    #fmax=FMAX, 
                                                    n_mels=N_MELS, 
                                                    power=POWER)
                #use log to get to human perceptive loudness levels
                mel_log = np.log(np.clip(mel, 1e-8, None))
                np.save(f"./data/mels/{i}-of-15/{filename}.npy", mel_log.astype(np.float32))
                counter += 1
                total_counter += 1

        print(f"Number of waveforms converted for file {i}: {counter}")

    print("Total Number of waveforms converted: ", total_counter)

def normalize_stats():
    mel_files = glob.glob("./data/mels/**/*.npy", recursive=True)

    sum_ = None
    sum_sq = None
    count = 0

    for f in tqdm(mel_files):
        mel = np.load(f)      # (80, T)
        mel = mel.T           # (T, 80)

        if sum_ is None:
            sum_ = np.zeros(mel.shape[1])
            sum_sq = np.zeros(mel.shape[1])

        sum_ += mel.sum(axis=0)            # sum over time
        sum_sq += (mel ** 2).sum(axis=0)
        count += mel.shape[0]

    mean = sum_ / count
    std = np.sqrt(sum_sq / count - mean**2)

    # safety clamp
    std = np.where(std < 1e-6, 1e-6, std)

    np.savez("mel_stats.npz", mean=mean, std=std)

    print("Saved mel_stats.npz")
    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)
    print("Mean range:", mean.min(), mean.max())
    print("Std range:", std.min(), std.max())


#convert_parquet_to_waveform()
convert_waveform_to_mel_spec()
normalize_stats() 