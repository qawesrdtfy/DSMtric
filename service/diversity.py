from tools.funcs import *
from config.Data import Data

def trig_class_diversity(data:Data) -> bool:
    """
    类别多样性的触发函数
    :param Y_modal: Y的模态
    :return: 是否触发，bool
    """
    if data.Y_modal==['类别']:
        return True
    return False
def class_diversity(data:Data):
    """
    类别多样性
    :param Y: 每个样本的类别
    :return: 类别多样性得分，范围0～1
    """
    probs=spread_probs(data.Y['类别'])
    score,therange=shannon_entropy(probs)
    score=zoom(score,therange[0],therange[1])
    return score


def trig_topic_diversity(data:Data) -> bool:
    """
    主题多样性的触发函数
    :param X_topic: 每个样本的主题
    :return: 是否触发，bool
    """
    if len(data.X_topic)==0:
        return False
    return True
def topic_diversity(data:Data):
    """
    主题多样性
    :param X_topics: 每个样本的主题
    :return: 主题多样性得分，范围0～1
    """
    probs=spread_probs(data.X_topic)
    score,therange=shannon_entropy(probs)
    score=zoom(score,therange[0],therange[1])
    return score


# 函数列表，元素为[指标名，触发函数，计算函数]
diversity_funclist=[["类别多样性",trig_class_diversity,class_diversity],
                    ["主题多样性",trig_topic_diversity,topic_diversity],]