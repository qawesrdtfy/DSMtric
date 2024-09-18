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
    统计含有偏见歧视的文本的占比
    """
    text = data.X['文本']
    res = ask_Discrimination(text)
    cnt = 0
    for it in res:
        if it == 'Yes':
            cnt += 1
    return round(cnt/len(text))


normative_funclist=[["偏见歧视",trig_discrimination_normative,discrimination_normative]]