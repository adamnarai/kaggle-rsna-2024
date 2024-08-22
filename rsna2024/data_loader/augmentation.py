import cv2
import logging

logging.getLogger('albumentations').setLevel(logging.WARNING)
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TransformFactory:
    def __init__(self):
        self.aug = None

    def get_transform(self):
        return self.aug


class NoAug(TransformFactory):
    def __init__(self):
        self.aug = A.Compose(
            [
                ToTensorV2(),
            ]
        )


class Affine(TransformFactory):
    def __init__(
        self,
        scale,
        translate_percent,
        rotate,
        shear,
        p=1.0,
    ):
        self.aug = A.Compose(
            [
                A.Affine(
                    scale=scale,
                    translate_percent=translate_percent,
                    rotate=rotate,
                    shear=shear,
                    p=p,
                ),
                ToTensorV2(),
            ]
        )


class CombinedV1(TransformFactory):
    def __init__(self, scale, translate_percent, rotate, shear, channel_shuffle_p=0.5, p=1.0):
        self.aug = A.Compose(
            [
                A.OneOf([A.Sharpen(p=0.5), A.MotionBlur(p=0.5)], p=0.5),
                A.ChannelShuffle(p=channel_shuffle_p),
                A.Affine(
                    scale=scale,
                    translate_percent=translate_percent,
                    rotate=rotate,
                    shear=shear,
                    p=p,
                ),
                ToTensorV2(),
            ]
        )
        
def ch_revert(image, **kwargs):
    return image[..., ::-1]

class CombinedChRevert(TransformFactory):
    def __init__(self, scale, translate_percent, rotate, shear, p=1.0):
        self.aug = A.Compose(
            [
                A.OneOf([A.Sharpen(p=0.5), A.MotionBlur(p=0.5)], p=0.5),
                A.Lambda(image=ch_revert, p=0.5),
                A.Affine(
                    scale=scale,
                    translate_percent=translate_percent,
                    rotate=rotate,
                    shear=shear,
                    p=p,
                ),
                ToTensorV2(),
            ]
        )
