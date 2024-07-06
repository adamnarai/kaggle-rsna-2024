import torch.nn as nn
import timm
import torch


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
    
class RSNASplit(nn.Module):
    def __init__(self, base_model, num_classes, pretrained=True, in_channels=None):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.model1 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=128, in_chans=in_channels[0])
        self.model2 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=128, in_chans=in_channels[1])
        self.model3 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=128, in_chans=in_channels[2])
        self.classifier = nn.Linear(3*128, self.num_classes)

    def forward(self, x1, x2, x3):
        x = torch.cat((self.model1(x1), self.model2(x2), self.model3(x3)), dim=1)
        return self.classifier(x)
