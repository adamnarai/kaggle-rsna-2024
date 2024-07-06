import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class TransformFactory():
    def __init__(self):
        self.aug = None

    def get_transform(self):
        return self.aug

class NoAug(TransformFactory):
    def __init__(self):
        self.aug = ToTensorV2()

class Elastic(TransformFactory):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, p=1.0):
        self.aug = A.Compose([A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, border_mode=cv2.BORDER_CONSTANT, p=p),
                              ToTensorV2()])

class Affine(TransformFactory):
    def __init__(self, p=1.0):
        self.aug = A.Compose([A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-10, 10), shear=(-10, 10), p=p),
                              ToTensorV2()])
