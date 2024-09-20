import outlines
import outlines.generate
import outlines.generate.choice
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from outlines.samplers import Sampler, greedy ,multinomial

class LoadLLM:
    def __init__(self,model_name) -> None:
        self.device = "cuda"
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = outlines.models.transformers(model_name,"auto",
            model_kwargs={
                "config": config,
                "trust_remote_code": True,
                "torch_dtype": "auto",
                "device_map": "auto",
            })
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sample = greedy()

    def chat_template(self,messages):
        """
        添加chat_template
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    
    def inter(self,messages:list):
        """
        与模型交互 --不做约束
        :messages: list形式如下
        messages = [
            {"role":"system","content":"You are a helpful assistant"},
            {"role":"user","content":"halo!"}
        ]

        :return:模型返回的文本
        """
        text_generator = outlines.generate.text(self.model,self.sample)
        text = self.chat_template(messages)
        response = text_generator(text,max_tokens=8192)
        return response
    
    def choice(self,messages:list,choices:list):
         """
         与模型交互 --约束为从choices中选择一个选项
         :choices: 模型回答内容的候选项
         :messages: fewshot和prompt
         :return: 模型的选择项
         """
         choice_generator = outlines.generate.choice(self.model,choices,self.sample)
         text = self.chat_template(messages)
         response = choice_generator(text,max_tokens=8192)
         return response

    def discrimination(self,text:list):
        """
        处理偏见歧视
        """
        Q = []
        for it in text:
            messages = [
                {"role":"system","content":"你的任务是判断用户的文本中是否包含歧视或偏见，如果是请输出Yes，否则输出No，除此之外不得输出任何多余内容。"},
                {"role":"user","content":it}
            ]
            Q.append(messages)
        return self.choice(Q,['Yes','No'])


    def valid(self,text:list):
        """
        处理是否符合现实逻辑
        """
        Q = []
        for it in text:
            messages = [
                {"role":"system","content":"你的任务是判断用户的文本中是否违背现实逻辑，如果违背现实逻辑请输出Yes，否则输出No，除此之外不得输出任何多余内容。"},
                {"role":"user","content":it}
            ]
            Q.append(messages)
        return self.choice(Q,['Yes','No'])

    def labelOK(self,rule,text:list):
        """
        处理是否符合标注规则
        """
        Q = []
        for it in text:
            messages = [
                {"role":"system","content":f"接下来会给你一个数据集及其标注结果，标注规则为：{rule}。你的任务是判断文本中的每个X对应的标注Y是否符合标注规则，如果符合上述标注规则输出Yes，否则输出No，除此之外不得输出任何多余内容。"},
                {"role":"user","content":f"X:{it['X']}\n标注结果Y:{it['Y']}"}
            ]
            Q.append(messages)
        return self.choice(Q,['Yes','No'])


if __name__ == "__main__":
    model = LoadLLM('/data/sdb2/lzy/LLM/Qwen2-7B-Instruct')
    text = model.discrimination(['黑人都是笨蛋','我们都有美好的未来'])
    print(text)
    text = model.valid(['鱼在天上飞','鸟在水底游泳','人在坐飞机'])
    print(text)
    text = model.labelOK('X为文本。Y为对应的X文本中是否含有错别字，Yes表明X中含有错别字，No表示X中不含有错别字。',[{"X":"身体健康","Y":"Yes"},{"X":"使用模型跑测试一遍数据集","Y":"No"},{"X":"我来自北京邮电大学","Y":"No"}])
    print(text)
