from config.Data import Data
from tools.askmodel import *
def trig_discrimination_normative(data:Data) -> bool:
    """
    文本规范性--偏见歧视检测触发函数
    """
    if data.X_modal==['文本']:
        return True
    return False
def discrimination_normative(data:Data):
    """
    文本规范性 --偏见歧视检测
    统计不含有偏见歧视的文本的占比
    """
    text = data.X['文本']
    res = ask_Discrimination(text)
    cnt = 0
    for it in res:
        if it == 'Yes':
            cnt += 1
    return round(cnt/len(text),4)

def LogicalLegality_normative(data:Data):
    """
    文本规范性 --现实逻辑错误检测
    统计不含有现实逻辑错误的文本的占比
    """
    text = data.X['文本']
    res = ask_LogicalLegality(text)
    cnt = 0
    for it in res:
        if it == 'Yes':
            cnt += 1
    return round(cnt/len(text),4)

def Guideline(data:Data):
    """"
    文本标注规则
    对标注质量进行评估，统计出标注质量较好的部分所占比例
    """
    textX = data.X['文本']
    textY = data.Y['文本']
    rule = data.rule

    data_list = []
    for i,item in enumerate(textX):
        data_list.append({"X":item,"Y":textY[i]})
    res = ask_Guideline(data_list,rule)
    # lzy快把这里补全啊！
    return 0.8


normative_funclist=[["偏见歧视",trig_discrimination_normative,discrimination_normative],
                    ["现实逻辑错误",trig_discrimination_normative,LogicalLegality_normative],
                    ["标注规则",trig_discrimination_normative,Guideline]]