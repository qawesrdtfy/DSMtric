# -*- encoding:utf -*-
import os
import json
from service.consistency import consistency_funclist
from service.diversity import diversity_funclist
from service.normative import normative_funclist

def trig(data):
    """
        判断指标是否可用。
        data：数据字典
        return：三类指标是否可用的布尔值列表
    """
    cons_trigged=[tfunc(data) for name,tfunc,_ in consistency_funclist]
    print('一致性指标可用性',cons_trigged)
    divers_trigged=[tfunc(data) for name,tfunc,_ in diversity_funclist]
    print('多样性指标可用性',cons_trigged)
    norma_trigged=[tfunc(data) for name,tfunc,_ in normative_funclist]
    print('多样性指标可用性',cons_trigged)
    return cons_trigged,divers_trigged,norma_trigged

def compute(data,cons_trigged,divers_trigged,norma_trigged):
    """
        计算指标。
        data：数据字典
        cons_trigged,divers_trigged,norma_trigged：三类指标是否可用的布尔值列表
        return：三类指标计算结果（放缩到0～1），如果指标不可用则为-1
    """
    cons_scores=[one[2](data) if cons_trigged[i] else -1 for i,one in enumerate(consistency_funclist)]
    divers_scores=[one[2](data) if divers_trigged[i] else -1 for i,one in enumerate(diversity_funclist)]
    norma_scores=[one[2](data) if norma_trigged[i] else -1 for i,one in enumerate(normative_funclist)]
    return cons_scores,divers_scores,norma_scores

def main(args):
    print('确定目录')
    dataset_dir=f'data/dataset/{args.username}-{args.datasetname}'
    result_dir=f'data/result/{args.username}-{args.datasetname}'
    print('合并数据集和元数据')
    X=read_X(dataset_dir)
    Y=read_Y(result_dir)
    data=json.loads(args.metadata)
    data['X']=X
    data['Y']=Y
    print('判断指标是否可用')
    cons_trigged, divers_trigged, norma_trigged=trig(data)
    print('计算每个指标的分数')
    cons_scores, divers_scores, norma_scores=compute(cons_trigged, divers_trigged, norma_trigged)
    print('计算每类指标的分数和总分')
    pure_cons_scores=[one for one in cons_scores if one!=-1]
    pure_divers_scores=[one for one in divers_scores if one!=-1]
    pure_norma_scores=[one for one in norma_scores if one!=-1]
    cons_score=round(sum(pure_cons_scores)/len(pure_cons_scores),4)
    divers_score=round(sum(pure_divers_scores)/len(pure_divers_scores),4)
    norma_score=round(sum(pure_norma_scores)/len(pure_norma_scores),4)
    final_score=round((cons_score+divers_score+norma_score)/3,4)
    # 保存结果和完成标识
    result={
        "总分":final_score,
        "一致性得分":cons_score,
        "多样性得分":divers_score,
        "规范性得分":norma_score
    }
    for i,item in enumerate(consistency_funclist):
        if cons_trigged[i]:
            result[item[0]]=cons_scores[i]
    for i,item in enumerate(diversity_funclist):
        if divers_trigged[i]:
            result[item[0]]=divers_scores[i]
    for i,item in enumerate(normative_funclist):
        if norma_trigged[i]:
            result[item[0]]=norma_scores[i]
    json.dump(result,open(result_dir+'/result.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(f'{args.username}的“{args.datasetname}”数据集评测完成')


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    parser.add_argument("--username", type=str)
    parser.add_argument("--datasetname", type=str)
    parser.add_argument("--metadata", type=str)
    args = parser.parse_args()
    main(args)

    