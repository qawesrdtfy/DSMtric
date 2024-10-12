# System.py的测试版本，区别在于，如果用户名数据集名与已有的重名，则会直接替代，而不是报错。（deal-dir的exist_ok置为True


import os
import subprocess
from flask import Flask,request,jsonify
import json
import shutil
import traceback

# 后端服务启动
app = Flask(__name__)

def deal_dir(username,datasetname,mode):
    """0建立1删除2检查是否存在"""
    dataset_dir=f'data/dataset/{username}-{datasetname}'
    result_dir=f'data/result/{username}-{datasetname}'
    if mode==0:
        os.makedirs(dataset_dir,exist_ok=True)
        os.makedirs(result_dir,exist_ok=True)
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
            print(request.form)
            print(request.form.to_dict())
            dataset = request.files['dataset']
            username = request.form.get("id")
            datasetname = request.form.get("name")
            metadata = request.form.get("metadata")
            print()
            print(username)
            print(datasetname)
            print(metadata)
            print()
            json.loads(metadata)
        except Exception as e:
            formResult = {"resultinfo":f'数据集评测启动失败，参数解析失败！{e}'}
            print('Abnormal Reponse:',formResult)
            traceback.print_exc()
            return jsonify(formResult)
        # 构建专属空间
        try:
            dataset_dir,result_dir,_=deal_dir(username,datasetname,mode=0)
        except Exception as e:
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测启动失败，存储空间构建失败！{e}'}
            print('Abnormal Reponse:',formResult)
            traceback.print_exc()
            return jsonify(formResult)
        # 存储并解压数据集
        try:
            # 貌似图像等会是好多个文件，所以需要人家发来个压缩包，然后咱保存了之后要解压
            # 要求解压后产生文件夹X和Y，X和Y内是模态为名的文件夹
            with open(f'{dataset_dir}/data.tar.gz', 'wb') as file:
                file.writelines(dataset.readlines())
            command = ["tar", "-xvf", f'data.tar.gz']
            process = subprocess.Popen(command, cwd=dataset_dir)
            # process = subprocess.Popen(command, stdout=file, stderr=file, cwd='.')
            process.wait()
        except Exception as e:
            deal_dir(username,datasetname,mode=1)
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测启动失败，数据集读取或存储失败！{e}'}
            print('Abnormal Reponse:',formResult)
            traceback.print_exc()
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
            traceback.print_exc()
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
        result_file=os.path.join(result_dir,'result.json')
        if not os.path.exists(result_file):
            formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测未完成！',"result":{}}
            print('Normal Reponse:',formResult)
            return jsonify(formResult)
        # 如果已完成
        result=json.load(open(result_file,'r',encoding='utf-8'))
        formResult = {"resultinfo":f'{username}的“{datasetname}”数据集评测完成！',"result":result}
        print('Normal Reponse:',formResult)
        return jsonify(formResult)

    return 'connection ok!'

if __name__ == '__main__':
    app.config['JSON_AS_ASCII']=False
    app.run(host='0.0.0.0',port=48814)
    print("GoodBye")
