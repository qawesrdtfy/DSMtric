from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class VLmodel:
    """"
    加载QwenVL模型
    """
    def __init__(self,modelname) -> None:
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(modelname, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(modelname)
        self.device = "cuda"
    
    def askmodel(self,prompt,imageurl):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": imageurl,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
if __name__ == "__main__":
    """
    for test
    """
    import jieba
    vLmodel = VLmodel('/data/sdb2/lzy/LLM/Qwen2-VL-7B-Instruct')
    ans = vLmodel.askmodel("请你用一句话描述一下图片的内容。注意！不需要出现类似于“这是一幅...”的描述。","https://c-ssl.duitang.com/uploads/blog/202304/03/20230403101857_d328e.jpg")
    from rouge_chinese import Rouge
    rouge = Rouge()
    print(ans)
    res = rouge.get_scores(' '.join(jieba.lcut(ans)),' '.join(jieba.lcut("四个活泼的动漫少女")))[0]['rouge-l']['f']
    print(res)
