# -*- encoding:utf -*-
import os
import json
from config.Data import Data
from service.consistency import consistency_funclist
from service.diversity import diversity_funclist
from service.normative import normative_funclist
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 设置时间格式
)
# 获取第三方库的日志记录器并设置日志级别
logging.getLogger("urllib3").setLevel(logging.WARNING)  # urllib3 是 requests 的依赖
logging.getLogger("requests").setLevel(logging.WARNING)  # 如果直接使用 requests
logging.getLogger("jieba").setLevel(logging.WARNING)  # 如果直接使用 requests

def trig(data):
    """
        判断指标是否可用。
        data：数据字典
        return：三类指标是否可用的布尔值列表
    """
    cons_trigged=[tfunc(data) for name,tfunc,_ in consistency_funclist]
    # print('\n[INFO] 一致性指标可用性')
    logging.info("\n\n一致性指标可用性")
    for i,item in enumerate(consistency_funclist):
        # print(f"{item[0]} : {cons_trigged[i]}")
        logging.info(f"{item[0]} : {cons_trigged[i]}")
    divers_trigged=[tfunc(data) for name,tfunc,_ in diversity_funclist]
    # print('\n[INFO] 多样性指标可用性')
    logging.info("\n\n多样性指标可用性")
    for i,item in enumerate(diversity_funclist):
        # print(f"{item[0]} : {divers_trigged[i]}")
        logging.info(f"{item[0]} : {divers_trigged[i]}")
    norma_trigged=[tfunc(data) for name,tfunc,_ in normative_funclist]
    # print('\n[INFO] 规范性指标可用性')
    logging.info("\n\n规范性指标可用性")
    for i,item in enumerate(normative_funclist):
        # print(f"{item[0]} : {norma_trigged[i]}")
        logging.info(f"{item[0]} : {norma_trigged[i]}")
    return cons_trigged,divers_trigged,norma_trigged

def get_level(num:float):
    """
    将数字转化为  [ABCDE等级制度, 五分制, 十分制, 百分制]
    """
    if num<0: return ["-","-","-"]
    ans=[]
    if num>=0.9:ans.append("A")
    elif num>=0.8:ans.append("B")
    elif num>=0.7:ans.append("C")
    elif num>=0.6:ans.append("D")
    else: ans.append("E")

    ans.append("{:.1f}".format(num * 5))
    ans.append("{:.1f}".format(num * 10))
    ans.append("{:.1f}".format(num * 100))

    return ans
    

def compute(data,cons_trigged,divers_trigged,norma_trigged):
    """
        计算指标。
        data：数据字典
        cons_trigged,divers_trigged,norma_trigged：三类指标是否可用的布尔值列表
        return：三类指标计算结果（放缩到0～1），如果指标不可用则为-1   和每个指标的名字
    """
    cons_scores=[]
    divers_scores=[]
    norma_scores=[]
    for i,one in enumerate(consistency_funclist):
        if cons_trigged[i]:
            C1 = time.time()
            # print("[INFO] " + consistency_funclist[i][0] + "指标开始评测")
            logging.info(f"{consistency_funclist[i][0]} 指标开始评测")
            cons_scores.append(one[2](data))
            # print("[INFO] " + consistency_funclist[i][0] + f"指标评测完成 总用时:{time.time()-C1}\n")
            logging.info(f"{consistency_funclist[i][0]} 指标评测完成 总用时:{time.time()-C1}\n")
        else:
            # print(f"[INFO] {consistency_funclist[i][0]} 指标未触发\n")
            logging.info(f"[INFO] {consistency_funclist[i][0]} 指标未触发\n")
            cons_scores.append(-1)

    for i,one in enumerate(diversity_funclist):
        if divers_trigged[i]:
            C1 = time.time()
            # print("[INFO] " + diversity_funclist[i][0] + "指标开始评测")
            logging.info(f"{diversity_funclist[i][0]} 指标开始评测")
            divers_scores.append(one[2](data))
            # print("[INFO] " + diversity_funclist[i][0] + f"指标评测完成 总用时:{time.time()-C1}\n")
            logging.info(f"{diversity_funclist[i][0]} 指标评测完成 总用时:{time.time()-C1}\n")
        else:
            # print(f"[INFO] {diversity_funclist[i][0]} 指标未触发\n")
            logging.info(f"[INFO] {diversity_funclist[i][0]} 指标未触发\n")
            divers_scores.append(-1)

    for i,one in enumerate(normative_funclist):
        if norma_trigged[i]:
            C1 = time.time()
            # print("[INFO] " + normative_funclist[i][0] + "指标开始评测")
            logging.info(f"{normative_funclist[i][0]} 指标开始评测")
            norma_scores.append(one[2](data))
            # print("[INFO] " + normative_funclist[i][0] + f"指标评测完成 总用时:{time.time()-C1}\n")
            logging.info(f"{normative_funclist[i][0]} 指标评测完成 总用时:{time.time()-C1}\n")
        else:
            # print(f"[INFO] {normative_funclist[i][0]} 指标未触发\n")
            logging.info(f"[INFO] {normative_funclist[i][0]} 指标未触发\n")
            norma_scores.append(-1)
    # cons_scores=[one[2](data) if cons_trigged[i] else -1 for i,one in enumerate(consistency_funclist)]
    # divers_scores=[one[2](data) if divers_trigged[i] else -1 for i,one in enumerate(diversity_funclist)]
    # norma_scores=[one[2](data) if norma_trigged[i] else -1 for i,one in enumerate(normative_funclist)]
    return cons_scores,divers_scores,norma_scores

def main(args):
    # print('确定目录')
    logging.info('确定目录')
    # 去除username
    dataset_dir=f'data/dataset/{args.datasetname}'
    result_dir=f'data/result/{args.datasetname}'
    weight = json.load(open('config/weight.json','r',encoding='utf-8'))
    taskdate = args.taskdate
    # print('合并数据集和元数据')
    logging.info('合并数据集合元数据')
    data=json.loads(args.metadata)
    data=Data(data, dataset_dir)
    # print('判断指标是否可用')
    logging.info('判断指标是否可用')
    cons_trigged, divers_trigged, norma_trigged=trig(data)
    # print('计算每个指标的分数')
    logging.info('计算每个指标的分数')
    cons_scores, divers_scores, norma_scores=compute(data, cons_trigged, divers_trigged, norma_trigged)
    # print(cons_scores,divers_scores,norma_scores)
    logging.info(cons_scores)
    logging.info(divers_scores)
    logging.info(norma_scores)
    # print('计算每类指标的分数和总分')
    logging.info('计算每类指标的分数和总分')
    pure_cons_scores=[one*weight[consistency_funclist[i][0]] for i,one in enumerate(cons_scores) if one!=-1]
    pure_divers_scores=[one*weight[diversity_funclist[i][0]] for i,one in enumerate(divers_scores) if one!=-1]
    pure_norma_scores=[one*weight[normative_funclist[i][0]] for i,one in enumerate(norma_scores) if one!=-1]

    pure_cons_scores_tot=[weight[consistency_funclist[i][0]] for i,one in enumerate(cons_scores) if one!=-1]
    pure_divers_scores_tot=[weight[diversity_funclist[i][0]] for i,one in enumerate(divers_scores) if one!=-1]
    pure_norma_scores_tot=[weight[normative_funclist[i][0]] for i,one in enumerate(norma_scores) if one!=-1]

    # print(pure_cons_scores)
    logging.info(pure_cons_scores)
    cons_score=round(sum(pure_cons_scores)/sum(pure_cons_scores_tot),4) if len(pure_cons_scores_tot)!=0 else -1
    # print(pure_divers_scores)
    logging.info(pure_divers_scores)
    divers_score=round(sum(pure_divers_scores)/sum(pure_divers_scores_tot),4) if len(pure_divers_scores_tot)!=0 else -1
    # print(pure_norma_scores)
    logging.info(pure_norma_scores)
    norma_score=round(sum(pure_norma_scores)/sum(pure_norma_scores_tot),4) if len(pure_norma_scores_tot)!=0 else -1
    three_scores=[one for one in [cons_score,divers_score,norma_score] if one!=-1]
    final_score=round(sum(three_scores)/len(three_scores),4) if len(three_scores)!=0 else -1
    # 保存结果和完成标识
    result={
        "总分":get_level(final_score),
        "一致性":{"总分":get_level(cons_score)},
        "多样性":{"总分":get_level(divers_score)},
        "规范性":{"总分":get_level(norma_score)}
    }
    for i,item in enumerate(consistency_funclist):
        if cons_trigged[i]:
            result["一致性"][item[0]]=get_level(cons_scores[i])
    for i,item in enumerate(diversity_funclist):
        if divers_trigged[i]:
            result["多样性"][item[0]]=get_level(divers_scores[i])
    for i,item in enumerate(normative_funclist):
        if norma_trigged[i]:
            result["规范性"][item[0]]=get_level(norma_scores[i])
    json.dump(result,open(result_dir+f'/{taskdate}/result.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    # print(f'“{args.datasetname}”数据集评测完成')
    logging.info(f'“{args.datasetname}”数据集评测完成')


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    # 去除username
    # parser.add_argument("--username", type=str)
    parser.add_argument("--datasetname", type=str)
    parser.add_argument("--metadata", type=str)
    parser.add_argument("--taskdate", type=str)
    args = parser.parse_args()
    main(args)

    