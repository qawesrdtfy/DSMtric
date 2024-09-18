import math
import numpy as np
import jieba
def zoom(x,min,max) -> float:
    """
    将取值放缩到0～1之间
    x: 原取值
    min：原取值的最小值
    max：原取值的最大值，要求大于min

    return: 放缩后的值，0~1
    """
    max-=min
    x-=min
    min-=min
    return x/max

def shannon_entropy(probs:list) -> float:
    """
    计算香农熵。
    probs: 每种取值的概率列表，元素范围(0,1]
    
    return: 香农熵的值，范围(0,log(n)]
    """
    count=0
    entropy = 0
    for p in probs:
        if p > 0:  # 因为0的对数是未定义的
            entropy -= p * math.log2(p)
            count+=1
    return entropy,(0,math.log2(count))

def spread_probs(labels:list) -> list:
    """
    计算每种取值的概率
    labels: 每个样本的取值
    
    return: 每种取值的概率列表（无序），元素范围(0,1]
    """
    length=len(labels)
    speads={}
    for label in labels:
        speads[label]=speads.get(label,0)+1
    for k,v in speads.items():
        speads[k]=v/length
    return list(speads.values())

def image_binary(matrix:np.ndarray) -> np.ndarray:
    """
    将图像转换为二值矩阵
    matrix：图像矩阵

    return：返回的二值图像矩阵
    """
    # 如果是彩色图像则首先转换为灰度图像
    image_gray = []
    if matrix.ndim == 3:
        R, G, B = matrix[:, :, 0], matrix[:, :, 1], matrix[:, :, 2]
        image_gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    else:
        image_gray = matrix
    # 灰度图像转换为二值矩阵
    binary_image = (image_gray > 127).astype(np.uint8)
    return binary_image

def calculate_ttr(tokens) -> float:
    """
    计算单个片段的Type-Token Ratio(TTR)
    tokens：文本词汇列表
    return：返回的TTR值
    """
    types=set(tokens)
    return  len(types)/len(tokens)

def batch_segment(texts, batch_size=500):
    """
    批量分词处理
    texts: 文本数据集 (list of str)
    batch_size: 每次处理的文本数量
    return: 生成器，逐批返回分词结果
    """
    batch_tokens = []
    for i, text in enumerate(texts):
        tokens = list(jieba.cut(text))
        batch_tokens.extend(tokens)

        if (i + 1) % batch_size == 0:
            yield batch_tokens  # 返回当前批次的所有分词结果
            batch_tokens = []  # 重置批次

    # 处理最后一个批次
    if batch_tokens:
        yield batch_tokens