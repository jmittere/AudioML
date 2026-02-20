#Preprocessing script to convert waveforms to log-mel spectrograms (using decibels)
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import soundfile as sf
import pandas as pd
import re

def decode_row(audio_dict):
    y, sr = librosa.load(io.BytesIO(audio_dict['bytes']),
                         sr=None, #keep same sample rate
                         mono=True) #convert to mono if stereo
    return y, sr

def sanitize(text):
    text = text.replace(" ", "")
    text = re.sub(r'[\\/*?:"<>|]', "", text)  #remove illegal chars
    return text

counter = 0
for i in range(0,15):
    
    #load all parquet files and convert them to their raw waveforms
    out_dir = f"./data/waveforms/{i}-of-15"
    os.makedirs(out_dir, exist_ok=True)

    if(i<10):
        df = pd.read_parquet(f'./data/raw_parquets/train-0000{i}-of-00015.parquet')
    else:
        df = pd.read_parquet(f'./data/raw_parquets/train-000{i}-of-00015.parquet')

    for idx, row in df.iterrows():
                try:
                    y, sr = decode_row(row['audio'])
                    title = sanitize(row['title'])
                    artist = sanitize(row['artist'])              
                    sf.write(f"{out_dir}/{idx}_{title}_{artist}.wav", y, sr, subtype="PCM_16")
                except Exception as e:
                    print(e)
                    print(f"{row['title']}, {row['artist']}, {row['language']}") 


    #Load the audio files from waveforms folder and convert to mel spectrograms (db)
    directory = f'./data/waveforms/{i}-of-15'
    out_dir = f'./data/mels/{i}-of-15'
    os.makedirs(out_dir, exist_ok=True)

    for entry in os.scandir(directory):  
        if entry.is_file():
            filename = entry.name.split(".wav")[0]
            samples, sample_rate = librosa.load(entry.path, sr=22050, mono=True) #consistent sampling rate and mono audio for all samples
            mel = librosa.feature.melspectrogram(y=samples,sr=sample_rate,n_fft=2048,hop_length=512,n_mels=128,power=2.0)
            #use decibel scale to get the final Mel Spectrogram
            mel_db = librosa.power_to_db(mel, ref=np.max)
            np.save(f"./data/mels/{i}-of-15/{filename}.npy", mel_db.astype(np.float32))
            counter += 1

print("Number of Files converted: ", counter)