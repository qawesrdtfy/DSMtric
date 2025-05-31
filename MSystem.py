import os
import subprocess
from flask import Flask,request,jsonify
import json
import torch
from model.models import *

# 后端服务启动
app = Flask(__name__)
config=json.load(open('config/config.json','r',encoding='utf-8'))
if config['DocEncoder']:
    docEncoder = DocEncoder('/data/sdb2/wyh/models/bert-base-chinese','cuda',128)
else:
    docEncoder = None
if config['VLmodel']:
    vLmodel = VLmodel('/data/sdb2/lzy/LLM/Qwen2-VL-2B-Instruct')
else:
    vLmodel = None
if config['PicEncoder']:
    picEncoder = PicEncoder('/data/sdb2/wyh/models/vit-base-patch16-224','cuda')
else:
    picEncoder = None
if config['Clip_Sim']:
    CLIPmode = Clip_Sim()
else:
    CLIPmode = None
if config['AudioEncoder']:
    audioEncoder = AudioEncoder('/data/sdb2/wyh/models/clap-htsat-unfused','cuda')
else:
    audioEncoder = None
if config['ASR']:
    ASRmodel = ASR('/data/sdb2/lzy/LLM/whisper-large-v3')
else:
    ASRmodel = None
if config['LoadLLM']:
    Qwen2 = LoadLLM('/data/sdb2/lzy/LLM/Qwen2.5-1.5B-Instruct')
else:
    Qwen2 = None
if config['MacBert']:
    SpellingCheck = MBert('/data/sdb2/lzy/LLM/macbert4csc-base-chinese')
else:
    SpellingCheck = None

if config['Inception']:
    Inception = InceptionModelV3('cuda')
else:
    Inception = None


@app.route("/doc_encode",methods=['post','get'])
def doc_encode():
    torch.cuda.empty_cache()
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        docs = data['docs']
        doc_encoded=docEncoder.encode(docs)
        formResult = {"resultinfo":doc_encoded}
        print('Normal Reponse:',"文档编码接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/pic2doc",methods=['post','get'])
def pic2doc():
    torch.cuda.empty_cache()
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        prompt = data['prompt']
        pic_paths = data['pic_paths']
        docs = [vLmodel.askmodel(prompt,one) for one in pic_paths]
        formResult = {"resultinfo":docs}
        print('Normal Reponse:',"图片生文档接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/pic_encode",methods=['post','get'])
def pic_encode():
    torch.cuda.empty_cache()
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        pic_paths = data['pic_paths']
        pic_encoded = picEncoder.encode(pic_paths)
        formResult = {"resultinfo":pic_encoded}
        print('Normal Reponse:',"图片编码接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/CLIPmodel",methods=['post','get'])
def CLIPmodel():
    torch.cuda.empty_cache()
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        pic_paths = data['pic_paths']
        text = data['text']
        # sim = [CLIPmode.calculate_similarity(item,text[i]) for i,item in enumerate(pic_paths)]
        sim = CLIPmode.calculate_similarity(pic_paths,text)
        formResult = {"resultinfo":sim}
        print('Normal Reponse:',"图文向量相似度接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/audio_encode",methods=['post','get'])
def audio_encode():
    torch.cuda.empty_cache()
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        audios = data['audios']
        audio_encoded = audioEncoder.encode(audios)
        formResult = {"resultinfo":audio_encoded}
        print('Normal Reponse:',"音频编码接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/audio2text",methods=['post','get'])
def audio2text():
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        audios = data['audios']
        audio_encoded = ASRmodel.Audio2text(audios)
        formResult = {"resultinfo":audio_encoded}
        print('Normal Reponse:',"音频转文本接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/discrimination",methods=['post','get'])
def discrimination():
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        text = data['text']
        response = Qwen2.discrimination(text)
        formResult = {"resultinfo":response}
        print('Normal Reponse:',"文本偏见歧视接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/valid",methods=['post','get'])
def valid():
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        text = data['text']
        response = Qwen2.valid(text)
        formResult = {"resultinfo":response}
        print('Normal Reponse:',"文本现实逻辑接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/guideline",methods=['post','get'])
def guideline():
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        text_pair = data['text_pair']
        rule = data['rule']
        response = Qwen2.labelOK(rule,text_pair)
        formResult = {"resultinfo":response}
        print('Normal Reponse:',"文本标注规则接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/WrongSpelling",methods=['post','get'])
def WrongSpelling():
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        text = data['text']
        response = SpellingCheck(text)
        formResult = {"resultinfo":response}
        print('Normal Reponse:',"中文错别字识别接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'

@app.route("/inceptionModel",methods=['post','get'])
def inceptionModel():
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        pic_paths = data['pic_paths']
        response =Inception.predict(pic_paths)
        formResult = {"resultinfo":response}
        print('Normal Reponse:',"Inception模型接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'


if __name__ == '__main__':
    app.config['JSON_AS_ASCII']=False
    app.run(host='0.0.0.0',port=48812)
    print("GoodBye")
