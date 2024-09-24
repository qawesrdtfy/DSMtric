from tools.funcs import *
from config.Data import Data
from collections import Counter
import librosa
from sklearn.metrics import pairwise_distances

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

def trig_length_diversity(data:Data) -> bool:
    """
    文本长度多样性的触发函数
    :param X_modal: 每个样本的模态，要求是文本
    :return: 是否触发，bool
    """
    if data.X_modal==["文本"]:
        return True
    return False

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
    if data.X_modal==["文本"]:
        return True
    return False

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
    if data.X_modal==["文本"]:
        return True
    return False

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


def trig_color_diversity(data:Data) -> bool:
    """
    颜色多样性的触发函数
    :param X_modal: 每个样本的模态，要求是图片
    :return: 是否触发，bool
    """
    if data.X_modal==["图像"]:
        return True
    return False

def color_diversity(data:Data):
    """
    计算图像的颜色多样性
    :param X: 每个样本，要求是图片
    :return: 熵值，取值范围为0-log(n*n*n)  n表示每个通道量化区间个数
    """
    images=data.X['图像']
    total_entropy = 0
    for image in images:
        color_hist = extract_color_histogram(image)
        image_entropy = entropy(color_hist)
        total_entropy += image_entropy
    return total_entropy / len(images)

def trig_visual_feature_diversity(data:Data):
    """
    视觉特征多样性的触发函数
    :param X_modal: 每个样本的模态，要求是视频
    :return: 是否触发，bool
    """
    if data.X_modal==["视频"]:
        return True
    return False

def visual_feature_diversity(data:Data):
    """
    视觉特征多样性
    :param X: 每个样本，要求是视频
    :return: 熵值，取值范围为0-log(n)  n是数据集所选取的视频帧总和
    """
    samples_frame=data.X["视频"]
    frame_interval=30  #视频提取间隔帧数 需要设置的指标
    #用于整体计算多个视频的整体指标
    all_color_features = []  #总体颜色特征
    all_shape_features = []  #总体形状特征
    all_texture_features = []  #总体纹理特征
    count=0
    for sample_frame in samples_frame:
        color_features = []
        shape_features = []
        texture_features = []
        for frame_count, frame in enumerate(sample_frame):
            if frame_count % frame_interval == 0:
                count=count+1
                color_hist = extract_color_histogram(frame)
                shape_feat = extract_shape_features(frame)
                texture_feat = extract_texture_features(frame)
                color_features.append(color_hist)
                shape_features.append(shape_feat)
                texture_features.append(texture_feat)
        all_color_features.extend(color_features)
        all_shape_features.extend(shape_features)
        all_texture_features.extend(texture_features)
    color_diversity = entropy(np.mean(all_color_features, axis=1))
    shape_diversity = entropy(all_shape_features)
    texture_diversity = np.mean([
                                entropy(np.array(all_texture_features)[:, 0]),
                                entropy(np.array(all_texture_features)[:, 1]),
                                entropy(np.array(all_texture_features)[:, 2])
                            ])

    score = np.mean([color_diversity, shape_diversity, texture_diversity])
    return score

def trig_audio_content_diversity(data:Data) -> bool:
    """
    内容特征多样性的触发函数
    :param X_modal: 每个样本的模态，要求是音频
    :return: 是否触发，bool
    """
    if data.X_modal==["音频"]:
        return True
    return False
def audio_content_diversity(data:Data):
    """
    音频内容多样性
    :param X: 每个样本，要求是音频
    :return: MFCC特征的余弦距离矩阵，取值范围为[0-2]  越高表示多样性越高
    """
    audio_samples=data.X['音频']
    audio_samples_file=data.X['音频地址']
    features=[]
    for audio_sample,audio_sample_file in zip(audio_samples,audio_samples_file):
        sr = librosa.get_samplerate(audio_sample_file)  # 获取采样率
        mfcc = librosa.feature.mfcc(y=audio_sample, sr=sr)
        features.append((mfcc.mean(axis=1)))
    dist_matrix = pairwise_distances(features, metric='cosine')
    score = np.mean(dist_matrix)
    return score

# 函数列表，元素为[指标名，触发函数，计算函数]
diversity_funclist=[["类别多样性",trig_class_diversity,class_diversity],
                    ["主题多样性",trig_topic_diversity,topic_diversity],
                    ["图像尺寸和比例多样性",trig_picShape_diversity,picShape_diversity],
                    ["视频长度和尺寸比例多样性",trig_videoLength_diversity,videoLength_diversity],
                    ["音频长度多样性",trig_audioLength_diversity,audioLength_diversity],
                    ["文本长度多样性", trig_length_diversity, length_diversity],
                    ["词汇量多样性", trig_vocabulary_diversity, vocabulary_diversity],
                    ["词汇丰富度", trig_vocabulary_richness, vocabulary_richness],
                    ["颜色多样性",trig_color_diversity,color_diversity],
                    ["视觉特征多样性",trig_visual_feature_diversity,visual_feature_diversity],
                    ["内容多样性",trig_audio_content_diversity,audio_content_diversity]
                    ]
