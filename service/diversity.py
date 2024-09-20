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

def trig_picShape_diversity(data:Data) -> bool:
    """
    图像尺寸和比例多样性的触发函数
    :param X_model: X的模态
    :return: 是否触发，bool
    """
    if data.X_modal == ["图像"]:
        return True
    return False
def picShape_diversity(data:Data):
    """
    图像尺寸和比例多样性
    :param X: 原始图像
    :return: 图像尺寸和比例多样性得分，范围0～1
    """
    pictures = data.X["图像地址"]
    shapes = []
    proportion = []
    for i,item in enumerate(pictures):
        shapes.append(item.shape)
        height = item.shape[0]
        width = item.shape[1]
        proportion.append(width/height)
    probsS = spread_probs(shapes)
    probsP = spread_probs(proportion)
    scoreS, therangeS = shannon_entropy(probsS)
    scoreP, therangeP = shannon_entropy(probsP)
    score1 = zoom(scoreS,therangeS[0],therangeS[1])
    score2 = zoom(scoreP,therangeP[0],therangeP[1])
    return (score1+score2)/2

def trig_videoLength_diversity(data:Data) -> bool:
    """
    视频长度和尺寸比例多样性的触发函数
    :param X_model: X的模态
    :return: 是否触发，bool
    """
    if data.X_modal == ["视频"]:
        return True
    return False
def videoLength_diversity(data:Data):
    """
    视频长度和尺寸比例多样性
    :param X: 视频数据
    :return: 视频长度和尺寸比例多样性得分，范围0～1
    """
    videos = data.X["视频"]
    length = []
    proportion = []
    for i,item in enumerate(videos):
        length.append(item.get_length())
        meta = item.metadata()
        width, height = meta['video_size']
        proportion.append(width/height)
    probsL = spread_probs(length)
    probsP = spread_probs(proportion)
    scoreL, therangeL = shannon_entropy(probsL)
    scoreP, therangeP = shannon_entropy(probsP)
    score1 = zoom(scoreL,therangeL[0],therangeL[1])
    score2 = zoom(scoreP,therangeP[0],therangeP[1])
    return (score1+score2)/2

def trig_audioLength_diversity(data:Data) -> bool:
    """
    音频长度多样性的触发函数
    :param X_model: X的模态
    :return: 是否触发，bool
    """
    if data.X_modal == ["音频"]:
        return True
    return False
def audioLength_diversity(data:Data):
    """
    音频长度多样性
    :param X: 音频数据
    :return: 音频长度多样性得分，范围0～1
    """
    audios = data.X["音频"]
    length = []
    for i,item in enumerate(audios):
        length.append(item.shape)
    probs = spread_probs(length)
    score, therange = shannon_entropy(probs)
    score = zoom(score,therange[0],therange[1])
    return score


# 函数列表，元素为[指标名，触发函数，计算函数]
diversity_funclist=[["类别多样性",trig_class_diversity,class_diversity],
                    ["主题多样性",trig_topic_diversity,topic_diversity],
                    ["图像尺寸和比例多样性",trig_picShape_diversity,picShape_diversity],
                    ["视频长度和尺寸比例多样性",trig_videoLength_diversity,videoLength_diversity],
                    ["音频长度多样性",trig_audioLength_diversity,audioLength_diversity]]