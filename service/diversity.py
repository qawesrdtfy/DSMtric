from ..tools.funcs import *

def trig_class_diversity(Y_modal:list) -> bool:
    """
    类别多样性的触发函数
    :param Y_modal: Y的模态
    :return: 是否触发，bool
    """
    if len(Y_modal)==1 and Y_modal[0]=='类别':
        return True
    return False
def class_diversity(Y:list):
    """
    类别多样性
    :param Y: 每个样本的类别
    :return: 类别多样性得分，范围0～1
    """
    probs=spread_probs(Y)
    score,therange=shannon_entropy(probs)
    score=zoom(score,therange[0],therange[1])
    return score

# 函数列表，元素为[指标名，触发函数，计算函数]
diversity_funclist=[["类别多样性",trig_class_diversity,class_diversity]]