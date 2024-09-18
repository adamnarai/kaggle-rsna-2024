import cv2
import logging

logging.getLogger('albumentations').setLevel(logging.WARNING)
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch


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


class CombinedV1Multiple(TransformFactory):
    def __init__(self, scale, translate_percent, rotate, shear, channel_shuffle_p, p):
        self.aug = []
        for i, _ in enumerate(scale):
            self.aug.append(
                A.Compose(
                    [
                        A.OneOf([A.Sharpen(p=0.5), A.MotionBlur(p=0.5)], p=0.5),
                        A.ChannelShuffle(p=channel_shuffle_p[i]),
                        A.Affine(
                            scale=scale[i],
                            translate_percent=translate_percent[i],
                            rotate=rotate[i],
                            shear=shear[i],
                            p=p[i],
                        ),
                        ToTensorV2(),
                    ]
                )
            )

def gaussian_heatmap(width, height, center, std_dev):
    """
    Args:
    - width (int): Width of the heatmap
    - height (int): Height of the heatmap
    - center (tuple): The (x, y) coordinates of the Gaussian peak
    - std_dev (int, optional): Standard deviation of the Gaussian

    """
    x_axis = torch.arange(width).float() - center[0]
    y_axis = torch.arange(height).float() - center[1]
    x, y = torch.meshgrid(y_axis, x_axis, indexing='ij')

    return torch.exp(-((x**2 + y**2) / (2 * std_dev**2)))


def center_gauss_masking(image, std_dev=15, **kwargs):
    heatmap = gaussian_heatmap(
        image.shape[1], image.shape[0], (image.shape[1] // 2, image.shape[0] // 2), std_dev
    ).numpy()
    return image * heatmap[...,None]


class CombinedV2(TransformFactory):
    def __init__(self, scale, translate_percent, rotate, shear, channel_shuffle_p=0.5, p=1.0):
        self.aug = A.Compose(
            [
                A.OneOf([A.Sharpen(p=0.5), A.MotionBlur(p=0.5)], p=0.5),
                A.ChannelShuffle(p=channel_shuffle_p),
                A.Lambda(image=center_gauss_masking, p=0.5),
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


class NoAugMultiple(TransformFactory):
    def __init__(self, dummy):
        self.aug = []
        for i, _ in enumerate(dummy):
            self.aug.append(
                A.Compose(
                    [
                        ToTensorV2(),
                    ]
                )
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
