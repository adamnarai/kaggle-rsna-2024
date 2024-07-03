# import albumentations as A
from torchvision.transforms import v2
import torch

class TransformFactory():
    def __init__(self):
        self.aug = None

    def get_transform(self):
        return self.aug

class NoAug(TransformFactory):
    def __init__(self):
        self.aug = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

class Elastic(TransformFactory):
    def __init__(self):
        self.aug = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                               v2.ElasticTransform()])

class Affine(TransformFactory):
    def __init__(self, p=0.5):
        self.aug = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                               v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)])
