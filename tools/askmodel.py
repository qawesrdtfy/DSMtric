import requests

def ask_DocEncoder(docs:list):
    r=requests.post('0.0.0.0:48812/doc_encode',json={"docs":docs})
    r=r.json()['resultinfo']
    return r

def ask_VLmodel(prompt:str, pic_paths:list):
    r=requests.post('0.0.0.0:48812/pic2doc',json={"prompt":prompt,"pic_paths":pic_paths})
    r=r.json()['resultinfo']
    return r

def ask_PicEncoder(pic_paths:list):
    r=requests.post('0.0.0.0:48812/pic_encode',json={"pic_paths":pic_paths})
    r=r.json()['resultinfo']
    return r

def ask_CLIPmodel(pic_paths:list,text:list):
    r=requests.post('0.0.0.0:48812/CLIPmodel',json={"pic_paths":pic_paths,"text":text})
    r=r.json()['resultinfo']
    return r