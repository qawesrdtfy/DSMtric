import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from rouge_chinese import Rouge
from skimage.metrics import structural_similarity as ssim
from ..tools.funcs import *
from ..config.Data import Data
rouge=Rouge()


def trig_class_consistency(data:Data) -> bool:
    """
    类别一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if len(data.Y_modal)==1 and data.Y_modal[0]=='类别':
        if len(data.Y_per_annotater)!=0:
            # 要求每个样本的标注员数量一样
            annotater_count=len(data.Y_per_annotater[0])
            for sample in data.Y_per_annotater:
                if annotater_count!=len(sample):
                    break
            else:
                return True
    return False
def class_consistency(data:Data):
    """
    类别一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 类别一致性得分，范围0～1
    """
    class_list=[]
    # 统计都有哪些类
    for sample in data.Y_per_annotater:
        for annotated in sample:
            if annotated not in class_list:
                class_list.append(annotated)
    # 统计每个样本每个类的支持人数
    class_statistic=[]
    for sample in data.Y_per_annotater:
        line=[0 for _ in class_list]
        for annotated in sample:
            line[class_list.index(annotated)]+=1
        class_statistic.append(line)
    class_statistic=np.array(class_statistic)
    # 计算fleiss_kappa
    kappa = fleiss_kappa(class_statistic, method='fleiss')
    kappa = 0 if kappa < 0 else kappa
    return kappa


def trig_content_consistency(data:Data) -> bool:
    """
    内容一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if len(data.Y_modal)==1 and data.Y_modal[0]=='文本':
        if len(data.Y_per_annotater)!=0:
            # 要求每个样本每个标注员标注的都是字符串类型
            for sample in data.Y_per_annotater:
                for per_annotate in sample:
                    if not isinstance(per_annotate,str):
                        break
                else:
                    continue
                break
            else:
                return True
    return False
def content_consistency(data:Data):
    """
    内容一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in data.Y_per_annotater:
        scores=[]
        sample_splited=[' '.join(list(one)) for one in sample]
        for i in range(len(sample_splited)):
            for j in range(i+1,len(sample_splited)):
                scores.append(rouge.get_scores(sample_splited[i],sample_splited[j])[0]['rouge-l']['f'])
        score=sum(scores)/len(scores)
        all_scores.append(score)
    return round(sum(all_scores)/len(all_scores),4)


def trig_visual_consistency(data:Data) -> bool:
    """
    视觉一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if len(data.Y_modal)==1 and data.Y_modal[0]=='图像':
        if len(data.Y_per_annotater)!=0:
            # 要求每个样本每个标注员标注的都是numpy矩阵类型，并且尺寸相同
            shape=data.Y_per_annotater[0][0].shape
            for sample in data.Y_per_annotater:
                for per_annotate in sample:
                    if not isinstance(per_annotate,np.ndarray) or per_annotate.shape!=shape:
                        break
                else:
                    continue
                break
            else:
                return True
        return True
    return False
def visual_consistency(data:Data):
    """
    视觉一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in data.Y_per_annotater:
        scores=[]
        for i in range(len(sample)):
            for j in range(i+1,len(sample)):
                scores.append(ssim(sample[i], sample[j], multichannel=True))
        score=sum(scores)/len(scores)
        all_scores.append(score)
    return round(sum(all_scores)/len(all_scores),4)

# 函数列表，元素为[指标名，触发函数，计算函数]
consistency_funclist=[["类别一致性",trig_class_consistency,class_consistency],
                    ["内容一致性",trig_content_consistency,content_consistency],
                    ["视觉一致性",trig_visual_consistency,visual_consistency],]