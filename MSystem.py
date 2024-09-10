import os
import subprocess
from flask import Flask,request,jsonify
import json
from model.models import *

docEncoder = DocEncoder('/data/sdb2/wyh/models/bert-base-chinese',128)
vLmodel = VLmodel('/data/sdb2/lzy/LLM/Qwen2-VL-7B-Instruct')
# 后端服务启动
app = Flask(__name__)


config=None
@app.route("/doc_encode",methods=['post','get'])
def doc_encode():
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
    if request.method == "POST":
        data = json.loads(request.get_data(as_text=True))
        prompt = data['prompt']
        pic_paths = data['pic_paths']
        docs = [vLmodel.askmodel(prompt,one) for one in pic_paths]
        formResult = {"resultinfo":docs}
        print('Normal Reponse:',"图片生文档接口调用成功")
        return jsonify(formResult)
    return 'connection ok!'


if __name__ == '__main__':
    app.config['JSON_AS_ASCII']=False
    app.run(host='0.0.0.0',port=48812)
    print("GoodBye")
