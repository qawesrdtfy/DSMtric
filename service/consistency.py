import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from rouge_chinese import Rouge
from ..tools.funcs import *
rouge=Rouge()


def trig_class_consistency(Y_modal=[], Y_per_annotater=[]) -> bool:
    """
    类别一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if len(Y_modal)==1 and Y_modal[0]=='类别' and len(Y_per_annotater)!=0:
        return True
    return False
def class_consistency(Y_per_annotater:list):
    """
    类别一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 类别一致性得分，范围0～1
    """
    class_list=[]
    # 统计都有哪些类
    for sample in Y_per_annotater:
        for annotated in sample:
            if annotated not in class_list:
                class_list.append(annotated)
    # 统计每个样本每个类的支持人数
    class_statistic=[]
    for sample in Y_per_annotater:
        line=[0 for _ in class_list]
        for annotated in sample:
            line[class_list.index(annotated)]+=1
        class_statistic.append(line)
    class_statistic=np.array(class_statistic)
    # 计算fleiss_kappa
    kappa = fleiss_kappa(class_statistic, method='fleiss')
    kappa = 0 if kappa < 0 else kappa
    return kappa


def trig_content_consistency(Y_modal=[], Y_per_annotater=[]) -> bool:
    """
    内容一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if len(Y_modal)==1 and Y_modal[0]=='文本' and len(Y_per_annotater)!=0:
        return True
    return False
def content_consistency(Y_per_annotater:list):
    """
    内容一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in Y_per_annotater:
        scores=[]
        sample_splited=[' '.join(list(one)) for one in sample]
        for i,annotated1 in enumerate(sample_splited):
            for j,annotated2 in enumerate(sample_splited):
                if i==j:
                    continue
                scores.append(rouge.get_scores(annotated1,annotated2)[0]['rouge-l']['f'])
        score=sum(scores)/len(scores)
        all_scores.append(score)
    return round(sum(all_scores)/len(all_scores),4)


# 函数列表，元素为[指标名，触发函数，计算函数]
diversity_funclist=[["类别一致性",trig_class_consistency,class_consistency],
                    ["主题多样性",trig_content_consistency,content_consistency],]