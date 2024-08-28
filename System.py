import os
import subprocess
from flask import Flask,request,jsonify
import json
import shutil

# 后端服务启动
app = Flask(__name__)

def deal_dir(username,datasetname,mode):
    """0建立1删除2检查是否存在"""
    dataset_dir=f'data/dataset/{username}-{datasetname}'
    result_dir=f'data/result/{username}-{datasetname}'
    if mode==0:
        os.makedirs(dataset_dir,exist_ok=False)
        os.makedirs(result_dir,exist_ok=False)
    elif mode==1:
        shutil.rmtree(dataset_dir)
        shutil.rmtree(result_dir)
    elif mode==2:
        pass
    else:
        raise Exception()
    return dataset_dir,result_dir, os.path.exists(dataset_dir) and os.path.exists(dataset_dir)

config=None
@app.route("/api/metric",methods=['post','get'])
def metric():
    if request.method == "POST":
        # 解析参数
        try:
            username = request.form.get("id")
            datasetname = request.form.get("name")
            metadata = request.form.get("metadata")
            json.loads(metadata)
            dataset = request.files['dataset']
        except Exception as e:
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测启动失败，参数解析失败！{e}'}
            print('Abnormal Reponse:',formResult)
            return jsonify(formResult)
        # 构建专属空间
        try:
            dataset_dir,result_dir,_=deal_dir(username,datasetname,mode=0)
        except Exception as e:
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测启动失败，存储空间构建失败！{e}'}
            print('Abnormal Reponse:',formResult)
            return jsonify(formResult)
        # 存储数据集
        try:
            with open(f'{dataset_dir}/data.save', 'wb') as file:
                file.writelines(dataset.readlines())
        except Exception as e:
            deal_dir(username,datasetname,mode=1)
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测启动失败，数据集读取或存储失败！{e}'}
            print('Abnormal Reponse:',formResult)
            return jsonify(formResult)
        # 启动评测
        try:
            with open(f'{result_dir}/metric.log', 'w') as file:
                # 创建子进程，将输出重定向到文件
                command = ["python", "ServiceMain.py", 
                        "--username", username,
                        "--metadata", metadata,
                        "--datasetname", datasetname]
                process = subprocess.Popen(command, stdout=file, stderr=file, cwd='.')
        except Exception as e:
            deal_dir(username,datasetname,mode=1)
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测启动失败，评测程序启动失败！{e}'}
            print('Abnormal Reponse:',formResult)
            return jsonify(formResult)

        formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测开始！'}
        print('Normal Reponse:',formResult)
        return jsonify(formResult)

    return 'connection ok!'

@app.route("/api/result",methods=['post','get'])
def result():
    if request.method == "POST":
        json_dict = request.get_data(as_text=True)
        json_dict = json.loads(json_dict)
        username = json_dict.get('id')
        datasetname = json_dict.get('name')

        # 检查是否存在目录
        dataset_dir,result_dir,exist=deal_dir(username,datasetname,mode=2)
        if not exist:
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集不存在！',"result":{}}
            print('Abnormal Reponse:',formResult)
            return jsonify(formResult)
        # 如果没完成
        if not os.path.exists(result_dir+'result.json'):
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测未完成！',"result":{}}
            print('Normal Reponse:',formResult)
            return jsonify(formResult)
        # 如果已完成
        result=json.load(open(result_dir+'result.json','r',encoding='utf-8'))
        formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测完成！',"result":result}
        print('Normal Reponse:',formResult)
        return jsonify(formResult)

    return 'connection ok!'

if __name__ == '__main__':
    app.config['JSON_AS_ASCII']=False
    app.run(host='0.0.0.0',port=48811)
    print("GoodBye")
