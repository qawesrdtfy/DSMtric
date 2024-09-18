from tools.funcs import *
from config.Data import Data
from collections import Counter

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

def trig_length_diversity(data:Data) -> bool:
    """
    文本长度多样性的触发函数
    :param X_modal: 每个样本的模态，要求是文本
    :return: 是否触发，bool
    """
    if len(data.X_modal)==["文本"]:
        return False
    return True

def length_diversity(data:Data):
    """
    文本长度多样性
    :param X: 每个样本的文本内容
    :return: 文本长度多样性得分 取值范围0-1
    """
    texts=data.X['文本']
    lengths = [len(text) for text in texts]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    score = std_length / mean_length
    return score

def trig_vocabulary_diversity(data:Data) -> bool:
    """
    词汇量的触发函数
    :param X_modal: 每个样本的模态，要求是文本
    :return: 是否触发，bool
    """
    if len(data.X_modal)==["文本"]:
        return False
    return True

def vocabulary_diversity(data:Data):
    """
    批量计算数据集的 STTR（标准化 Type-Token Ratio）
    :param X: 每个样本的文本
    :return: 数据集的 STTR 值  取值范围0-1
    """
    texts=data.X['文本']
    total_ttr_values = []
    total_segments = 0
    #batch_size 预设值 500
    #segment_length 预设值 500
    segment_length=500
    batch_size=500
    for batch_tokens in batch_segment(texts, batch_size=batch_size):
        segments = [batch_tokens[i:i + segment_length] for i in range(0, len(batch_tokens), segment_length)]
        ttr_values = [calculate_ttr(segment) for segment in segments]
        total_ttr_values.extend(ttr_values)
        total_segments += len(ttr_values)

    # 计算并返回 STTR
    return sum(total_ttr_values) / total_segments if total_segments > 0 else 0


def trig_vocabulary_richness(data:Data) ->bool:
    """
    词汇丰富度的触发函数
    :param X_modal: 每个样本的模态，要求是文本
    :return: 是否触发，bool
    """
    if len(data.X_modal)==["文本"]:
        return False
    return True

def vocabulary_richness(data:Data):
    """
    计算文本的基尼系数 其思想是将词频分布类比为收入分布，于是可以计算基尼系数
    :param X: 每个样本的文本
    :return: 数据集的基尼系数  取值范围0-1  越接近0表示越均衡
    """
    texts=data.X['文本']
    global_word_counts = Counter()
    # 对每个文本分别进行分词并统计词频
    for text in texts:
        words = jieba.cut(text)
        word_counts = Counter(words)
        global_word_counts.update(word_counts)
    # Step 2: 计算全局基尼系数
    frequencies = np.array(list(global_word_counts.values()))
    score = compute_gini(frequencies)
    return score


# 函数列表，元素为[指标名，触发函数，计算函数]
diversity_funclist=[["类别多样性",trig_class_diversity,class_diversity],
                    ["主题多样性",trig_topic_diversity,topic_diversity],
                    ["文本长度多样性", trig_length_diversity, length_diversity],
                    ["词汇量多样性", trig_vocabulary_diversity, vocabulary_diversity]
                    ]