import numpy as np
from transformers import AutoTokenizer, BertModel
class DocEncoder:
    def __init__(self, modelpath, device, max_length) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.model = BertModel.from_pretrained(modelpath).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

    def encode(self, docs:list) -> list:
        tknzd = self.tokenizer.batch_encode_plus(docs,add_special_tokens=False,padding='max_length',truncation=True,max_length=self.max_length,return_tensors='pt').to(self.device)
        encoded=self.model(tknzd['input_ids'],tknzd['attention_mask'])['pooler_output']
        return encoded.cpu().tolist()


if __name__=='__main__':
    c=DocEncoder('/data/sdb2/wyh/models/bert-base-chinese','cuda',32)
    r=c.encode(['我吃了个包子','你今晚吃的什么'])
    print(len(r))
    print(r[0].shape)