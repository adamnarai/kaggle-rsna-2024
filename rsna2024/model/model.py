import torch.nn as nn
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


class TilesSagt2Model(nn.Module):
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


class SplitCoordModel(nn.Module):
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
            classes=self.num_classes[0],
            in_channels=self.in_channels[0],
            encoder_weights=self.encoder_weights,
        )

    def forward(self, x):
        return self.unet(x)


class RSNAMilSplit(nn.Module):
    def __init__(
        self,
        base_model,
        num_classes,
        pretrained=True,
        in_channels=None,
        resolution=None,
        instance_num=None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.resolution = resolution
        self.instance_num = instance_num
        self.encoder_classes = 128
        self.model1 = timm.create_model(
            model_name=self.base_model,
            pretrained=pretrained,
            num_classes=self.encoder_classes,
            in_chans=in_channels[0],
        )
        self.model2 = timm.create_model(
            model_name=self.base_model,
            pretrained=pretrained,
            num_classes=self.encoder_classes,
            in_chans=in_channels[1],
        )
        self.model3 = timm.create_model(
            model_name=self.base_model,
            pretrained=pretrained,
            num_classes=self.encoder_classes,
            in_chans=in_channels[2],
        )
        self.maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.maxpool3 = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(3 * self.encoder_classes, self.num_classes)

    def forward(self, x1, x2, x3):
        x1 = x1.view(
            -1, self.in_channels[0], *x1.shape[-2:]
        )  # (bs*instance_num, in_channels, H, W)
        x2 = x2.view(-1, self.in_channels[1], *x2.shape[-2:])
        x3 = x3.view(-1, self.in_channels[2], *x3.shape[-2:])
        feature1 = (
            self.model1(x1).view(-1, self.instance_num[0], self.encoder_classes).swapaxes(1, 2)
        )  # (bs, encoder_classes, N)
        feature2 = (
            self.model2(x2).view(-1, self.instance_num[1], self.encoder_classes).swapaxes(1, 2)
        )
        feature3 = (
            self.model3(x3).view(-1, self.instance_num[2], self.encoder_classes).swapaxes(1, 2)
        )
        feature1 = self.maxpool1(feature1).squeeze(-1)  # (bs, encoder_classes)
        feature2 = self.maxpool2(feature2).squeeze(-1)
        feature3 = self.maxpool3(feature3).squeeze(-1)
        
        x = torch.cat((feature1, feature2, feature3), dim=1)
        return self.classifier(x)
