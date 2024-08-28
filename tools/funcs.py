import math

def zoom(x,min,max) -> float:
    """
    将取值放缩到0～1之间
    x: 原取值
    min：原取值的最小值
    max：原取值的最大值，要求大于min

    return: 放缩后的值，0~1
    """
    return x/(max-min)

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
    return speads