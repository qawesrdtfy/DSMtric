import math
import numpy as np
import jieba
from collections import Counter
from scipy.stats import entropy
import cv2
from skimage.feature import graycomatrix, graycoprops


def zoom(x, min, max) -> float:
    """
    将取值放缩到0～1之间
    x: 原取值
    min：原取值的最小值
    max：原取值的最大值，要求大于min

    return: 放缩后的值，0~1
    """
    if abs(max-min) < 1e-6:
        return 0
    max -= min
    x -= min
    min -= min
    return round(x/max, 4)


def shannon_entropy(probs: list) -> float:
    """
    计算香农熵。
    probs: 每种取值的概率列表，元素范围(0,1]

    return: 香农熵的值，范围(0,log(n)]
    """
    count = 0
    entropy = 0
    for p in probs:
        if p > 0:  # 因为0的对数是未定义的
            entropy -= p * math.log2(p)
            count += 1
    return entropy, (0, math.log2(count))


def spread_probs(labels: list) -> list:
    """
    计算每种取值的概率
    labels: 每个样本的取值

    return: 每种取值的概率列表（无序），元素范围(0,1]
    """
    length = len(labels)
    speads = {}
    for label in labels:
        speads[label] = speads.get(label, 0)+1
    for k, v in speads.items():
        speads[k] = v/length
    return list(speads.values())


def image_binary(matrix: np.ndarray) -> np.ndarray:
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
    types = set(tokens)
    return len(types)/len(tokens)


def batch_segment(texts, batch_size=500):
    """
    批量分词处理
    texts: 文本数据集
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


def compute_gini(frequencies):
    """
    计算基尼系数
    frequencies: 词频

    return: 基尼系数，取值范围 0-1
    """
    # 将词频按升序排列
    frequencies = np.sort(frequencies)
    n = len(frequencies)
    # 计算累积频率的比例
    cumulative_freqs = np.cumsum(frequencies) / sum(frequencies)
    # 计算基尼系数
    gini_index = 1 - 2 * np.sum(cumulative_freqs) / n

    return gini_index


def quantize_colors(image, bins=16):
    """
    量化颜色，把原来复杂的颜色空间量化到指定的bins区间，减少颜色数量
    image: 图像的RGB矩阵
    bins: 量化的区间个数

    return: 量化后的图像矩阵
    """
    img_quantized = image // (256 // bins)
    return img_quantized


def compute_color_frequencies(image, bins=256):
    """
    计算每个图像的颜色分布频率
    image: 图像的RGB矩阵
    bins: 量化的区间个数

    return: 图像矩阵在每个区间的频率统计
    """
    img_quantized = quantize_colors(image, bins)
    pixels = img_quantized.reshape(-1, 3)
    pixels_tuple = [tuple(pixel) for pixel in pixels]
    color_counter = Counter(pixels_tuple)
    return color_counter


def compute_image_entropy(color_counter):
    """
    计算图像颜色熵值
    color_counter: 图像矩阵在每个区间的频率统计

    return: 熵值
    """
    total_pixels = sum(color_counter.values())
    # 计算每种颜色出现的概率
    probabilities = np.array(list(color_counter.values())) / total_pixels
    return entropy(probabilities, base=2)


def extract_color_histogram(image, n_bins):
    """
    提取颜色直方图作为颜色特征
    image: 图像的RGB矩阵

    return: 一维数组，表示RGB三个通道的颜色分布
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [n_bins, n_bins, n_bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_shape_features(image):
    """
    提取形状特征 (使用边缘检测)
    image: 图像的RGB矩阵

    return: 边缘数量，作为形状特征
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges)

#


def extract_texture_features(image):
    """
    提取灰度共生矩阵 (GLCM) 的纹理特征
    image: 图像的RGB矩阵

    return: contrast 对比度, dissimilarity 不相似度, homogeneity 均匀度
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]  # 值越高，图像对比度越明显  取值 0-无穷
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]  # 值越高，图像中相邻像素的灰度差异越大
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]  # 值越高，图像中相邻像素的灰度值越相似。
    return contrast, dissimilarity, homogeneity
