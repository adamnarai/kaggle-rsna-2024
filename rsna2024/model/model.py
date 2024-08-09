import torch.nn as nn
from torch.nn import functional as F
import logging
import timm
import torch
import segmentation_models_pytorch as smp

logging.getLogger('timm').setLevel(logging.WARNING)


class BaselineModel(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        return self.model(x)


class SplitModel(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.model1 = timm.create_model(
            model_name=self.base_model,
            pretrained=pretrained,
            num_classes=128,
            in_chans=in_channels[0],
        )
        self.model2 = timm.create_model(
            model_name=self.base_model,
            pretrained=pretrained,
            num_classes=128,
            in_chans=in_channels[1],
        )
        self.model3 = timm.create_model(
            model_name=self.base_model,
            pretrained=pretrained,
            num_classes=128,
            in_chans=in_channels[2],
        )
        self.classifier = nn.Linear(3 * 128, self.num_classes)

    def forward(self, x1, x2, x3):
        x = torch.cat((self.model1(x1), self.model2(x2), self.model3(x3)), dim=1)
        return self.classifier(x)


class CoordModel(nn.Module):
    def __init__(
        self,
        base_model,
        encoder_name,
        num_classes=None,
        in_channels=None,
        encoder_weights='imagenet',
    ):
        super().__init__()
        self.base_model = base_model
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encoder_weights = encoder_weights

        self.unet = getattr(smp, self.base_model)(
            encoder_name=self.encoder_name,
            classes=self.num_classes,
            in_channels=self.in_channels,
            encoder_weights=self.encoder_weights,
        )

    def forward(self, x):
        return self.unet(x)


class SpinalROIModel(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.model_sagt2 = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            num_classes=128,
            in_chans=in_channels,
        )
        self.classifier = nn.Linear(128 + 5, num_classes)

    def forward(self, x_sagt2, level):
        x_sagt2 = self.model_sagt2(x_sagt2)
        x = self.classifier(torch.cat((x_sagt2, level), dim=1))
        return x


class ForaminalROIModel(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.model_sagt1 = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            num_classes=128,
            in_chans=in_channels,
        )
        self.classifier = nn.Linear(128 + 5, num_classes)

    def forward(self, x_sagt1, level):
        x_sagt1 = self.model_sagt1(x_sagt1)
        x = self.classifier(torch.cat((x_sagt1, level), dim=1))
        return x


class SubarticularROIModel(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.model_sagt2 = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            num_classes=128,
            in_chans=in_channels,
        )
        self.model_axi = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            num_classes=128,
            in_chans=in_channels,
        )
        self.classifier = nn.Linear(128 + 128, num_classes)

    def forward(self, x_axi, x_sagt2, level):
        x_sagt2 = self.model_sagt2(x_sagt2)
        x_axi = self.model_axi(x_axi)
        x = self.classifier(torch.cat((x_axi, x_sagt2), dim=1))
        return x
