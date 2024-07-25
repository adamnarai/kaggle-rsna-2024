import cv2
import logging

logging.getLogger('albumentations').setLevel(logging.WARNING)
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    def __init__(self, scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-20, 20), shear=(-20, 20), p=1.0):
        self.aug = A.Compose([A.Affine(scale=scale, translate_percent=translate_percent, rotate=rotate, shear=shear, p=p),
                              ToTensorV2()])
        
class AffineCoord(TransformFactory):
    def __init__(self, scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-20, 20), shear=(-20, 20), p=1.0):
        self.aug = A.Compose([A.Affine(scale=scale, translate_percent=translate_percent, rotate=rotate, shear=shear, p=p),
                            #   A.Normalize(mean=0.0, std=1.0),
                              ToTensorV2()])
