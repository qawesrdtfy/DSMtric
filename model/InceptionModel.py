import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import numpy as np

class InceptionModelV3:
    def __init__(self,device):
        # 加载Inception v3模型，使用预训练的参数
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.eval()  # 设置为评估模式
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image):
        """对输入的图片进行预处理"""
        return self.preprocess(image).unsqueeze(0)

    def predict(self, images):
        """
        对输入的图片列表进行预测
        :param images: PIL Image列表
        :return: 预测结果（概率分布）
        """
        preds = []
        for img in images:
            input_tensor = self.preprocess_image(img)
            with torch.no_grad():
                pred = self.model(input_tensor)
                preds.append(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)
if __name__ == "__main__":
    image_paths = ["testdata/image/image1.jpg", "testdata/image/image2.jpg"]  # 替换为实际的图片路径
    images = [Image.open(image_path) for image_path in image_paths]

    # 创建InceptionModel对象
    model = InceptionModelV3()

    # 获取预测结果
    predictions = model.predict(images)

    # 打印预测结果
    print("Predictions:")
    for i, pred in enumerate(predictions):
        print(f"Image {i + 1}: {pred}")