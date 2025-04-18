import os
import numpy as np
import pandas as pd
import skimage.io
import skvideo.io
import librosa
import json

filesize=json.load(open("config/filesize_config.json","r",encoding='utf-8'))

class FileTooLargeError(Exception):
    def __init__(self, file_size, max_size):
        self.file_size = file_size
        self.max_size = max_size
        super().__init__(f"文件大小 {file_size}B 超过管理员限制 {max_size}B")
    
def checksize(files,modal):
    MAXSIZE=filesize[modal]*1024
    for file in files:
        # print(file)
        file_size=os.path.getsize(file)
        if(file_size>MAXSIZE):
            raise FileTooLargeError(file_size, MAXSIZE)

def read_structure(path):  # -> list[np.ndarray]
    files = [os.path.join(path, one) for one in os.listdir(path)]
    files.sort()
    checksize(files,'csv')
    data = pd.read_csv(files[0]).to_numpy()
    samples = [line for line in data]
    return samples


def read_str(path):  # -> list[str]
    files = [os.path.join(path, one) for one in os.listdir(path)]
    files.sort()
    if len(files) == 1:
        samples = [line.strip('\n') for line in open(files[0], 'r', encoding='utf-8').readlines() if line != '\n']
    else:
        samples = ['\n'.join(open(file, 'r', encoding='utf-8').readlines()) for file in files]
    for i in range(len(samples)):
        MAXLENGTH=filesize['text-length']
        if(len(samples[i])>MAXLENGTH):
            samples[i]=samples[i][:MAXLENGTH]
            print("[Warning] 文本过长，已进行裁切。")
    return samples


def read_picture(path):  # -> list[np.ndarray]
    files = [os.path.join(path, one) for one in os.listdir(path)]
    files.sort()
    checksize(files,'image')
    samples = [skimage.io.imread(file) for file in files]
    return samples


def split_audio_to_fixed_frames(y, frame_length, hop_length):
    """
    将音频分割成固定长度的帧
    :param y: 音频信号
    :param frame_length: 每帧的长度（样本数）
    :param hop_length: 帧移（样本数）
    :return: 分割后的帧列表
    """
    frames = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
        frames.append(frame)
    frames=np.vstack(frames)
    frames=np.mean(frames,axis=1)
    return frames

def read_audio(path):  # -> list[np.ndarray]
    files = [os.path.join(path, one) for one in os.listdir(path)]
    files.sort()
    checksize(files,'audio')
    samples = []
    for file in files:
        y,sr=librosa.load(file, sr=None)
        y=split_audio_to_fixed_frames(y,sr//2,sr//8)
        samples.append(y)
    return samples


def read_video(path):  # -> list[np.ndarray]
    import numpy
    numpy.float = numpy.float64
    numpy.int = numpy.int_
    import skvideo.io
    files = [os.path.join(path, one) for one in os.listdir(path)]
    files.sort()
    checksize(files,'video')
    samples = [skvideo.io.vread(file) for file in files]
    return samples
