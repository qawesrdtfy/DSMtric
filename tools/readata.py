import os
import numpy as np
import pandas as pd
import skimage.io
import skvideo.io
import librosa

def read_structure(path): # -> list[np.ndarray]
    files=[os.path.join(path,one) for one in os.listdir(path)]
    data=pd.read_csv(files[0]).to_numpy()
    samples=[line for line in data]
    return samples

def read_str(path): #  -> list[str]
    files=[os.path.join(path,one) for one in os.listdir(path)]
    if len(files)==1:
        samples=[line.strip('\n') for line in open(files[0],'r',encoding='utf-8').readlines()]
    else:
        samples=[open(file,'r',encoding='utf-8').readlines() for file in files]
    return samples

def read_picture(path): #  -> list[np.ndarray]
    files=[os.path.join(path,one) for one in os.listdir(path)]
    samples=[skimage.io.imread(file) for file in files]
    return samples

def read_audio(path): #  -> list[np.ndarray]
    files=[os.path.join(path,one) for one in os.listdir(path)]
    samples=[librosa.load(file,sr=None)[0] for file in files]
    return samples

def read_video(path): #  -> list[np.ndarray]
    files=[os.path.join(path,one) for one in os.listdir(path)]
    samples=[skvideo.io.vread(file) for file in files]
    return samples
