"""
任务：测试指标合理性，每个人测试自己负责的指标的输出结果是否合理
合理性定义：如果你根据指标计算的原理，认为这个数据集的这个指标得分应当是（趋近于）0，就需要检查指标计算实际输出是否满足。
具体步骤：
1、对自己负责的每一个指标：
    1.1、修改import的函数为自己写的指标触发函数和指标计算函数
    1.2、修改metadata内容为指标所需的模态
    1.3、把对应的数据集上传到data/dataset/test01下面
    1.4、如果指标用到模型，详见最下面的绿色注释
    1.5、运行本文件，看看指标计算输出得分是否与预料的一样
    1.5、重复1.1~1.4步，分别测试指标计算结果理应趋近于0、1、正常的情况。
"""


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
