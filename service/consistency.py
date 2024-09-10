import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from rouge_chinese import Rouge
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from ..tools.funcs import *
from ..config.Data import Data
import jieba
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


def trig_docontent_consistency(data:Data) -> bool:
    """
    文本内容一致性的触发函数
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
def docontent_consistency(data:Data):
    """
    文本内容一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in data.Y_per_annotater:
        scores=[]
        sample_splited=[' '.join(jieba.lcut(item)) for item in sample]
        for i in range(len(sample_splited)):
            for j in range(i+1,len(sample_splited)):
                scores.append(rouge.get_scores(sample_splited[i],sample_splited[j])[0]['rouge-l']['f'])
        score=sum(scores)/len(scores)
        all_scores.append(score)
    return round(sum(all_scores)/len(all_scores),4)


def image_text_consistancy(data:Data):
    '''
    图文内容一致性  --图像转文本模型，rougel相似度
    :Y_paths
    :Y
    :return:图文内容一致性得分，范围0～1
    '''
    if len(data.X_) != len(data.Y): # 这个写到触发函数里
        raise ValueError("两个列表的长度不相同")
    
    all_scores=[]
    for i,item in enumerate(X_images):
        text = QwenVL.askmodel("请你用简短的语言描述一下图片的内容。",item)
        textx = ' '.join(jieba.lcut(text))
        texty = ' '.join(jieba.lcut(Y[i]))
        all_scores.append(rouge.get_scores(textx,texty)[0]['rouge-l']['f'])
    return round(sum(all_scores)/len(all_scores),4)
    

def trig_docfeature_consistency(data:Data) -> bool:
    """
    文本向量特征一致性的触发函数
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
def docfeature_consistency(data:Data):
    """
    文本向量特征一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :param model: 文本编码模型
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in data.Y_per_annotater:
        encoded_sample=model.encode(sample)
        cos_matric=cosine_similarity(encoded_sample)
        score=np.sum(np.fill_diagonal(cos_matric,0))
        all_scores.append(score)
    final_score=round(sum(all_scores)/len(all_scores),4)
    return zoom(final_score,-1,1)


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


def image_text_consistancy(X_images = [], Y = []):
    '''
    图文内容一致性  --图像转文本模型，rougel相似度
    :X_images: 图片url列表
    :Y:图片描述
    :return:图文内容一致性得分，范围0～1
    '''
    if len(X_images) != len(Y):
        raise ValueError("两个列表的长度不相同")
    
    all_scores=[]
    for i,item in enumerate(X_images):
        text = QwenVL.askmodel("请你用简短的语言描述一下图片的内容。",item)
        textx = ' '.join(jieba.lcut(text))
        texty = ' '.join(jieba.lcut(Y[i]))
        all_scores.append(rouge.get_scores(textx,texty)[0]['rouge-l']['f'])
    return round(sum(all_scores)/len(all_scores),4)
    


# 函数列表，元素为[指标名，触发函数，计算函数]
consistency_funclist=[["类别一致性",trig_class_consistency,class_consistency],
                    ["文本内容一致性",trig_docontent_consistency,docontent_consistency],
                    ["文本向量特征一致性",trig_docfeature_consistency,docfeature_consistency],
                    ["视觉一致性",trig_visual_consistency,visual_consistency],]