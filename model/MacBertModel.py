from pycorrector.macbert.macbert_corrector import MacBertCorrector


class MBert:
    def __init__(self,modelpath):
        self.model = MacBertCorrector(modelpath).correct_batch

    def __call__(self,data:list):
        res = self.model(data,threshold=0.5)
        ans = [0 if len(it['errors'])==0 else 1 for it in res]
        return ans


if __name__ == '__main__':
    test = MBert("/data/sdb2/lzy/LLM/macbert4csc-base-chinese")
    
    ans = test(['你是水？','你是个大大大大好人','李治毅','评审采用建康安全的材质，单手即可打开的按压式杯盖设计','北京邮电大大学'])

    print(ans)
    # [0, 0, 0, 1, 0]
