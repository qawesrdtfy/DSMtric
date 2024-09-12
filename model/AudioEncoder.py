from transformers import ClapModel, ClapProcessor
class AudioEncoder:
    def __init__(self, modelpath, device) -> None:
        self.model = ClapModel.from_pretrained(modelpath).to(device)
        self.processor = ClapProcessor.from_pretrained(modelpath)
        self.model.eval()
        self.device = device

    def encode(self, audios:list) -> list:
        inputs = self.processor(audios=audios, return_tensors="pt").to(self.device)
        audio_embed = self.model.get_audio_features(**inputs)
        return audio_embed.cpu().tolist()

if __name__=='__main__':
    import numpy as np
    x=np.random.randn(20).tolist()
    c=AudioEncoder('/data/sdb2/wyh/models/clap-htsat-unfused','cuda')
    r=c.encode(x)
    print(r)