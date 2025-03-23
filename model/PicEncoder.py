from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

class PicEncoder:
    def __init__(self, modelpath, device) -> None:
        self.processor = ViTImageProcessor.from_pretrained(modelpath)
        self.model = ViTForImageClassification.from_pretrained(modelpath).to(device)
        self.model.eval()
        self.device = device

    def encode(self, pic_paths:list) -> list:
        images=[Image.open(open(path,'rb')).convert('RGB') for path in pic_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        return logits.cpu().tolist()


if __name__=='__main__':
    c=PicEncoder('/data/sdb2/wyh/models/vit-base-patch16-224','cuda')
    r=c.encode(['image.png'])
    print(r)