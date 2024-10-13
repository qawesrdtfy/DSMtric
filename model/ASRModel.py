from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class ASR:
    def __init__(self, model_id) -> None:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype="auto", low_cpu_mem_usage=True, use_safetensors=True, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype="auto",
            device_map="auto",
        )

    def Audio2text(self, AudioURL):
        ans = self.pipe(AudioURL)
        res = ans['text']
        return res


if __name__ == "__main__":
    test = ASR('/data/sdb2/lzy/LLM/whisper-large-v3').Audio2text(
        'data/dataset/超级派蒙旋风先生-THUNEWS/X/音频/3.wav')
    print(test)
