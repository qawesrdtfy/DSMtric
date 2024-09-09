import os
import numpy as np
import pandas as pd
import skimage.io
import skvideo.io
import librosa
from ..config.Data import Data

def read_structure(path) -> list[np.ndarray]:
    files=[os.path.join(path,one) for one in os.listdir(path)]
    data=pd.read_csv(files[0]).to_numpy()
    samples=[line for line in data]
    return samples

def read_str(path) -> list[str]:
    files=[os.path.join(path,one) for one in os.listdir(path)]
    if len(files)==1:
        samples=[line.strip('\n') for line in open(files[0],'r',encoding='utf-8').readlines()]
    else:
        samples=[open(file,'r',encoding='utf-8').readlines() for file in files]
    return samples

def read_picture(path) -> list[np.ndarray]:
    files=[os.path.join(path,one) for one in os.listdir(path)]
    samples=[skimage.io.imread(file) for file in files]
    return samples

def read_audio(path) -> list[np.ndarray]:
    files=[os.path.join(path,one) for one in os.listdir(path)]
    samples=[librosa.load(file,sr=None)[0] for file in files]
    return samples

def read_video(path) -> list[np.ndarray]:
    files=[os.path.join(path,one) for one in os.listdir(path)]
    samples=[skvideo.io.vread(file) for file in files]
    return samples

modal2func={
    "结构化数据":read_structure,
    "文本":read_str,
    "类别":read_str,
    "图像":read_picture,
    "音频":read_audio,
    "语音":read_audio,
    "视频":read_video
}

def read_XY(dataset_dir,data:Data):
    X_path=os.path.join(dataset_dir,'X')
    X={}
    for modal in data.X_modal:
        X[modal]=modal2func[modal](os.path.join(X_path,modal))
    X=modal2func[data.X_modal](X_path)
    Y_path=os.path.join(dataset_dir,'Y')
    Y=modal2func[data.Y_modal](Y_path)
    return X,Y