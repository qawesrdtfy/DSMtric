import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from rouge_chinese import Rouge
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr,spearmanr
from librosa.feature import mfcc
from dtaidistance import dtw
from tools.funcs import *
from tools.askmodel import *
from config.Data import Data
import jieba
rouge=Rouge()


def trig_class_consistency(data:Data) -> bool:
    """
    类别一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if data.Y_modal==['类别']:
        if len(data.Y_per_annotater['类别'])!=0:
            # 要求每个样本的标注员数量一样
            annotater_count=len(data.Y_per_annotater['类别'][0])
            for sample in data.Y_per_annotater['类别']:
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
    for sample in data.Y_per_annotater['类别']:
        for annotated in sample:
            if annotated not in class_list:
                class_list.append(annotated)
    # 统计每个样本每个类的支持人数
    class_statistic=[]
    for sample in data.Y_per_annotater['类别']:
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
    if data.Y_modal==['文本']:
        if len(data.Y_per_annotater['文本'])!=0:
            # 要求每个样本每个标注员标注的都是字符串类型
            for sample in data.Y_per_annotater['文本']:
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
    for sample in data.Y_per_annotater['文本']:
        scores=[]
        sample_splited=[' '.join(jieba.lcut(item)) for item in sample]
        for i in range(len(sample_splited)):
            for j in range(i+1,len(sample_splited)):
                scores.append(rouge.get_scores(sample_splited[i],sample_splited[j])[0]['rouge-l']['f'])
        score=sum(scores)/len(scores)
        all_scores.append(score)
    return round(sum(all_scores)/len(all_scores),4)



def trig_audiocontent_consistency(data:Data) -> bool:
    """
    音频内容一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if data.Y_modal==['音频'] and len(data.Y_per_annotater['音频'])!=0:
        return True
    return False
def audiocontent_consistency(data:Data):
    """
    音频内容一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    maxone=0
    for sample in data.Y_per_annotater['音频']:
        scores=[]
        mfcc_sample = [mfcc(one) for one in sample]
        for i in range(len(mfcc_sample)):
            for j in range(i+1,len(mfcc_sample)):
                d=dtw(mfcc_sample[i],mfcc_sample[j])
                maxone=d if d>maxone else maxone
                scores.append(d)
        score=sum(scores)/len(scores)
        all_scores.append(score)
    final_score=round(sum(all_scores)/len(all_scores),4)
    return 1-zoom(final_score,0,maxone)


def trig_image_text_consistancy(data:Data):
    '''
    图文内容一致性的触发函数  --图像转文本模型，rougel相似度
    :Y_pic_paths
    :Y_modal
    :return:bool
    '''
    if len(data.Y_modal)==2 and '文本' in data.Y_modal and '图像' in data.Y_modal:
        return True
    return False
def image_text_consistancy(data:Data):
    '''
    图文内容一致性  --图像转文本模型，rougel相似度
    :Y_pic_paths
    :Y['文本']
    :return:图文内容一致性得分，范围0～1
    '''
    all_scores=[]
    for i,item in enumerate(data.Y['图像地址']):
        text = ask_VLmodel("请你用简短的语言描述一下图片的内容。",[item])[0]
        textx = ' '.join(jieba.lcut(text))
        texty = ' '.join(jieba.lcut(data.Y['文本'][i]))
        all_scores.append(rouge.get_scores(textx,texty)[0]['rouge-l']['f'])
    return round(sum(all_scores)/len(all_scores),4)
def image_text_vec_consistancy(data:Data):
    '''
    图文向量特征一致性  --CLIP
    :Y_pic_paths
    :Y['文本']
    :return:图文内容一致性得分，范围0～1
    '''
    all_scores = []
    for i,item in enumerate(data.Y_pic_paths):
        sim = ask_CLIPmodel(item,data.Y['文本'][i])
        all_scores.append(sim)
    return round(sum(all_scores)/len(all_scores),4)
        

def trig_docfeature_consistency(data:Data) -> bool:
    """
    文本向量特征一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if data.Y_modal==['文本']:
        if len(data.Y_per_annotater['文本'])!=0:
            # 要求每个样本每个标注员标注的都是字符串类型
            for sample in data.Y_per_annotater['文本']:
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
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in data.Y_per_annotater['文本']:
        encoded_sample=ask_DocEncoder(sample)
        cos_matric=cosine_similarity(encoded_sample)
        score=np.sum(np.fill_diagonal(cos_matric,0)) / (len(sample)**2-len(sample))
        all_scores.append(score)
    final_score=round(sum(all_scores)/len(all_scores),4)
    return zoom(final_score,-1,1)


def trig_audiofeature_consistency(data:Data) -> bool:
    """
    音频向量特征一致性的触发函数
    :param Y_modal: Y的模态
    :return: 是否触发，bool
    """
    if data.Y_modal==['音频'] and len(data.Y_per_annotater['音频'])!=0:
        return True
    return False
def audiofeature_consistency(data:Data):
    """
    音频向量特征一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in data.Y_per_annotater['音频']:
        encoded_sample=ask_AudioEncoder(sample)
        cos_matric=cosine_similarity(encoded_sample)
        score=np.sum(np.fill_diagonal(cos_matric,0)) / (len(sample)**2-len(sample))
        all_scores.append(score)
    final_score=round(sum(all_scores)/len(all_scores),4)
    return zoom(final_score,-1,1)


def trig_picfeature_consistency(data:Data) -> bool:
    """
    图像向量特征一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if data.Y_modal==['图像']:
        return True
    return False
def picfeature_consistency(data:Data):
    """
    图像向量特征一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 内容一致性得分，范围0～1
    """
    all_scores=[]
    for sample in data.Y_per_annotater['图像地址']:
        encoded_sample=ask_PicEncoder(sample)
        cos_matric=cosine_similarity(encoded_sample)
        score=np.sum(np.fill_diagonal(cos_matric,0)) / (len(sample)**2-len(sample))
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
    if data.Y_modal==['图像']:
        if len(data.Y_per_annotater['图像'])!=0:
            # 要求每个样本每个标注员标注的都是numpy矩阵类型，并且尺寸相同
            shape=data.Y_per_annotater['图像'][0][0].shape
            for sample in data.Y_per_annotater['图像']:
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
    for sample in data.Y_per_annotater['图像']:
        scores=[]
        for i in range(len(sample)):
            for j in range(i+1,len(sample)):
                scores.append(ssim(sample[i], sample[j], multichannel=True))
        score=sum(scores)/len(scores)
        all_scores.append(score)
    return round(sum(all_scores)/len(all_scores),4)


def trig_goals_consistency(data:Data) -> bool:
    """
    目标一致性的触发函数
    :param Y_modal: Y的模态
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 是否触发，bool
    """
    if data.Y_modal==['图像']:
        if len(data.Y_per_annotater['图像'])!=0:
            # 要求每个样本每个标注员标注的都是numpy矩阵类型，并且尺寸相同
            shape=data.Y_per_annotater['图像'][0][0].shape
            for sample in data.Y_per_annotater['图像']:
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
def goals_consistency(data:Data):
    """
    目标一致性
    :param Y_per_annotater: 每个样本每个标注员的标注结果
    :return: 目标一致性得分，范围0～1
    """
    all_scores = []
    # 计算jaccard index需要先将每个标注员标注的numpy矩阵转换为二值矩阵(只有元素0和1)
    for sample in data.Y_per_annotater['图像']:
        scores = []
        for i in range(len(sample)):
            for j in range(i+1,len(sample)):
                scores.append(jaccard_score(image_binary(i), image_binary(j), average="samples"))
                score=sum(scores)/len(scores)
        all_scores.append(score)
    return round(sum(all_scores)/len(all_scores),4)


def trig_person_consistency(data:Data) -> bool:
    """
    线性相关一致性的触发函数
    :param Y_modal: Y的模态
    :return: 是否触发，bool
    """
    modals=['文本','音频','图像'] # TODO 后期还会有语音、视频，当然需要对应修改下面的计算函数
    if len(data.Y_modal)==1 and data.Y_modal[0] in modals and \
    len(data.X_modal)==1 and data.X_modal[0] in modals:
        return True
    return False
def person_consistency(data:Data):
    """
    线性相关一致性
    :param X: 
    :param Y: 
    :return: 线性相关一致性得分，范围0～1
    """
    modal2func={
        "文本":ask_DocEncoder,
        "音频":ask_AudioEncoder,
        "图像":ask_PicEncoder
    }
    X_embeded = modal2func[data.X_modal[0]](data.X[data.X_modal[0]])
    Y_embeded = modal2func[data.Y_modal[0]](data.Y[data.Y_modal[0]])
    X_cosmatrix = cosine_similarity(X_embeded).flatten().tolist()
    Y_cosmatrix = cosine_similarity(Y_embeded).flatten().tolist()
    person,_ = pearsonr(X_cosmatrix,Y_cosmatrix)
    return zoom(abs(person),-1,1)


def trig_spearman_consistency(data:Data) -> bool:
    """
    非线性相关一致性的触发函数
    :param Y_modal: Y的模态
    :return: 是否触发，bool
    """
    modals=['文本','音频','图像'] # TODO 后期还会有语音、视频，当然需要对应修改下面的计算函数
    if len(data.Y_modal)==1 and data.Y_modal[0] in modals and \
    len(data.X_modal)==1 and data.X_modal[0] in modals:
        return True
    return False
def spearman_consistency(data:Data):
    """
    非线性相关一致性
    :param X: 
    :param Y: 
    :return: 线性相关一致性得分，范围0～1
    """
    modal2func={
        "文本":ask_DocEncoder,
        "音频":ask_AudioEncoder,
        "图像":ask_PicEncoder
    }
    X_embeded = modal2func[data.X_modal[0]](data.X[data.X_modal[0]])
    Y_embeded = modal2func[data.Y_modal[0]](data.Y[data.Y_modal[0]])
    X_cosmatrix = cosine_similarity(X_embeded).flatten().tolist()
    Y_cosmatrix = cosine_similarity(Y_embeded).flatten().tolist()
    spearmanr,_ = spearmanr(X_cosmatrix,Y_cosmatrix)
    return zoom(abs(spearmanr),-1,1)

# 函数列表，元素为[指标名，触发函数，计算函数]
consistency_funclist=[["类别一致性",trig_class_consistency,class_consistency],
                    ["文本内容一致性",trig_docontent_consistency,docontent_consistency],
                    ["音频内容一致性",trig_audiocontent_consistency,audiocontent_consistency],
                    ["文本向量特征一致性",trig_docfeature_consistency,docfeature_consistency],
                    ["图像向量特征一致性",trig_picfeature_consistency,picfeature_consistency],
                    ["音频向量特征一致性",trig_audiofeature_consistency,audiofeature_consistency],
                    ["图文内容一致性",trig_image_text_consistancy,image_text_consistancy],
                    ["视觉一致性",trig_visual_consistency,visual_consistency],
                    ["线性相关一致性",trig_person_consistency,person_consistency],
                    ["非线性相关一致性",trig_spearman_consistency,spearman_consistency],
                    ["目标一致性",trig_goals_consistency,goals_consistency]]
