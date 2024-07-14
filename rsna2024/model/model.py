import torch.nn as nn
import logging
import timm
import torch

logging.getLogger('timm').setLevel(logging.WARNING)

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
    
class RSNASplitCoord(nn.Module):
    def __init__(self, base_model, num_classes, pretrained=True, in_channels=None):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.model1 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=128, in_chans=in_channels[0])
        self.model2 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=128, in_chans=in_channels[1])
        self.model3 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=128, in_chans=in_channels[2])
        self.classifier = nn.Linear(3*128, self.num_classes)
        self.regressor = nn.Linear(3*128, 50)

    def forward(self, x1, x2, x3):
        x = torch.cat((self.model1(x1), self.model2(x2), self.model3(x3)), dim=1)
        return self.classifier(x), self.regressor(x)
    

class RSNAMilSplit(nn.Module):
    def __init__(self, base_model, num_classes, pretrained=True, in_channels=None, resolution=None, instance_num=None):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.resolution = resolution
        self.instance_num = instance_num
        self.encoder_classes = 128
        self.model1 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=self.encoder_classes, in_chans=in_channels[0])
        self.model2 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=self.encoder_classes, in_chans=in_channels[1])
        self.model3 = timm.create_model(model_name=self.base_model, pretrained=pretrained, num_classes=self.encoder_classes, in_chans=in_channels[2])
        # self.rnn1 = nn.GRU(self.encoder_classes, self.encoder_classes, 2, dropout=0.5, batch_first=True, bidirectional=False)
        # self.rnn2 = nn.GRU(self.encoder_classes, self.encoder_classes, 2, dropout=0.5, batch_first=True, bidirectional=False)
        # self.rnn3 = nn.GRU(self.encoder_classes, self.encoder_classes, 2, dropout=0.5, batch_first=True, bidirectional=False)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.maxpool3 = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(3*self.encoder_classes, self.num_classes)

    def forward(self, x1, x2, x3):
        x1 = x1.view(-1, self.in_channels[0], *x1.shape[-2:])  # (bs*instance_num, in_channels, H, W)
        x2 = x2.view(-1, self.in_channels[1], *x2.shape[-2:])
        x3 = x3.view(-1, self.in_channels[2], *x3.shape[-2:])
        feature1 = self.model1(x1).view(-1, self.instance_num[0], self.encoder_classes).swapaxes(1, 2)  # (bs, encoder_classes, N)
        feature2 = self.model2(x2).view(-1, self.instance_num[1], self.encoder_classes).swapaxes(1, 2)
        feature3 = self.model3(x3).view(-1, self.instance_num[2], self.encoder_classes).swapaxes(1, 2)
        feature1 = self.maxpool1(feature1).squeeze(-1)  # (bs, encoder_classes)
        feature2 = self.maxpool2(feature2).squeeze(-1)
        feature3 = self.maxpool3(feature3).squeeze(-1)
        # feature1 = self.model1(x1).view(-1, self.instance_num[0]*self.encoder_classes)  # (bs, N*encoder_classes)
        # feature2 = self.model2(x2).view(-1, self.instance_num[1]*self.encoder_classes)
        # feature3 = self.model3(x3).view(-1, self.instance_num[2]*self.encoder_classes)
        # feature1 = self.rnn1(feature1)[0].reshape(-1, self.instance_num*self.encoder_classes)  # (bs, N*encoder_classes)
        # feature2 = self.rnn2(feature2)[0].reshape(-1, self.instance_num*self.encoder_classes)
        # feature3 = self.rnn3(feature3)[0].reshape(-1, self.instance_num*self.encoder_classes)
        x = torch.cat((feature1, feature2, feature3), dim=1)
        return self.classifier(x)

    