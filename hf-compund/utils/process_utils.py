import numpy as np
from torchvision import transforms

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])
def build_preprocess(image_shape1,image_shape2):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_shape1, image_shape2)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 常用标准化
    ])
    return preprocess
def build_postprocess():
    postprocess = transforms.Compose([
        lambda x: (x / 2 + 0.5).clamp(0, 1),
        transforms.ToPILImage()
    ])
    return postprocess
