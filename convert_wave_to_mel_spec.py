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
            df = pd.read_parquet(f'./data/raw_parquets/train-0000{i}-of-00015.parquet')
        else:
            df = pd.read_parquet(f'./data/raw_parquets/train-000{i}-of-00015.parquet')

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
    for i in range(0,15):
        counter = 0
        directory = f'./data/waveforms/{i}-of-15'
        out_dir = f'./data/mels/{i}-of-15'
        os.makedirs(out_dir, exist_ok=True)

        for entry in os.scandir(directory):  
            if entry.is_file():
                filename = entry.name.split(".wav")[0]
                samples, sample_rate = librosa.load(entry.path, sr=22050, mono=True) #consistent sampling rate and mono audio for all samples
                #params needed for HIFI-GAN vocoder for output post processing
                mel = librosa.feature.melspectrogram(y=samples,
                                                    sr=sample_rate,
                                                    n_fft=1024,
                                                    hop_length=256,
                                                    win_length=1024,
                                                    fmin=0.0, 
                                                    fmax=8000.0, 
                                                    n_mels=80, 
                                                    power=1.0)
                #use log to get to human perceptive loudness levels
                mel_log = np.log(np.clip(mel, 1e-5, None))
                np.save(f"./data/mels/{i}-of-15/{filename}.npy", mel_log.astype(np.float32))
                counter += 1
                total_counter += 1

        print(f"Number of waveforms converted for file {i}: {counter}")

    print("Total Number of waveforms converted: ", total_counter)

convert_waveform_to_mel_spec()