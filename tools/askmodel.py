import requests

def ask_DocEncoder(docs:list, B=10):
    rets=[]
    for i in range(0,len(docs),B):
        r=requests.post('http://0.0.0.0:48812/doc_encode',json={"docs":docs[i:i+B]})
        r=r.json()['resultinfo']
        rets+=r
    return rets

def ask_VLmodel(prompt:str, pic_paths:list, B=10):
    rets=[]
    for i in range(0,len(pic_paths),B):
        r=requests.post('http://0.0.0.0:48812/pic2doc',json={"prompt":prompt,"pic_paths":pic_paths[i:i+B]})
        r=r.json()['resultinfo']
        rets+=r
    return rets

def ask_PicEncoder(pic_paths:list, B=10):
    rets=[]
    for i in range(0,len(pic_paths),B):
        r=requests.post('http://0.0.0.0:48812/pic_encode',json={"pic_paths":pic_paths[i:i+B]})
        r=r.json()['resultinfo']
        rets+=r
    return rets

def ask_CLIPmodel(pic_paths:list,text:list, B=10):
    rets=[]
    for i in range(0,len(pic_paths),B):
        r=requests.post('http://0.0.0.0:48812/CLIPmodel',json={"pic_paths":pic_paths[i:i+B],"text":text[i:i+B]})
        r=r.json()['resultinfo']
        rets+=r
    return rets

def ask_AudioEncoder(audios:list, B=10):
    rets=[]
    for i in range(0,len(audios),B):
        r=requests.post('http://0.0.0.0:48812/audio_encode',json={"audios":audios[i:i+B]})
        r=r.json()['resultinfo']
        rets+=r
    return rets

def ask_WhisperModel(audios:list):
    r=requests.post('0.0.0.0:48812/audio2text',json={"audios":audios})
    r=r.json()['resultinfo']
    return r