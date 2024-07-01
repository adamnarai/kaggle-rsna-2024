import torch.nn as nn
import timm


class RSNABaseline(nn.Module):
    def __init__(self, base_model, num_classes, pretrained=True, in_channels=None):
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
