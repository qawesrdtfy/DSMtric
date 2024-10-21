
from config.Data import Data
# 以下需要修改：引入触发函数和计算函数。修改为所需函数、所需模态
from service.consistency import trig_docfeature_consistency as trigge_func
from service.consistency import docfeature_consistency as func
metadata = {"X_modal": ["文本"], "Y_modal": ["文本"]}

# 以下无需修改
dataset_dir = 'data/dataset/test01'  # 数据集文件放到这里
data = Data(metadata, dataset_dir)
trigged = trigge_func(data)
if trigged:
    score = func(data)
    print("得分", score)
else:
    print("未触发")

# 如果指标计算函数需要使用模型：去config/config.json将所需模型置为true，其他模型为false，然后启动MSystem.py。（如果启动时报环境问题，则conda activate /data/sdb2/lzy/Miniconda3/envs/DSMtric，再启动
# 数据集文件格式
"""
data/dataset/test01
-X
--模态1名称
---模态1数据文件
--模态2名称
---模态2数据文件
-Y
--模态1名称
---模态1数据文件
--模态2名称
---模态2数据文件
-Y_per_annotater
--1
---模态1名称
----模态1数据文件
--2
---模态2名称
----模态2数据文件
"""
