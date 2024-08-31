import torch.nn as nn
from torch.nn import functional as F
import logging
import timm
import torch
import segmentation_models_pytorch as smp

logging.getLogger('timm').setLevel(logging.WARNING)

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


class SplitROIFeatures(nn.Module):
    def __init__(
        self,
        base_model,
        num_classes,
        in_channels=None,
        pretrained=True,
        rnn_hidden_size=512,
        rnn_num_layers=2,
        rnn_dropout=0,
        rnn_bidirectional=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            in_chans=self.in_channels,
        )
        self.feature_num = self.model.num_features
        self.global_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.GRU(
            input_size=self.feature_num,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )
        if rnn_bidirectional:
            fc_in_size = rnn_hidden_size * 2
        else:
            fc_in_size = rnn_hidden_size
        self.classifier = nn.Linear(fc_in_size + 5, num_classes)

    def forward(self, x):
        x_subset_list = []
        start_size = self.in_channels // 2
        end_size = self.in_channels - start_size
        for i in range(start_size, x.shape[1] - end_size + 1):
            x_subset = x[:, i - start_size : i + end_size, :, :]
            x_subset = self.model.forward_features(x_subset)
            x_subset = self.global_pool_2d(x_subset).flatten(start_dim=1)
            x_subset_list.append(x_subset)
        x_split = torch.stack(x_subset_list, dim=1)

        # RNN
        x_split, _ = self.rnn(x_split)
        x_split = x_split.permute(0, 2, 1)
        x_split = self.global_pool_1d(x_split).squeeze()
        if x_split.dim() == 1:
            x_split = x_split.unsqueeze(0)

        return x_split


class SpinalROIModelV2(nn.Module):
    def __init__(
        self,
        base_model,
        num_classes,
        in_channels=None,
        pretrained=True,
        rnn_hidden_size=512,
        rnn_num_layers=2,
        rnn_dropout=0,
        rnn_bidirectional=False,
    ):
        # TODO: future remove
        if isinstance(in_channels, int):
            in_channels = [5, 1]
        
        super().__init__()
        self.model_sagt2 = SplitROIFeatures(
            base_model,
            num_classes,
            in_channels=in_channels[0],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_axi = SplitROIFeatures(
            base_model,
            num_classes,
            in_channels=in_channels[1],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.classifier = nn.Linear(2 * rnn_hidden_size, num_classes)

    def forward(self, x_sagt2, x_axi, level):
        x_sagt2 = self.model_sagt2(x_sagt2)
        x_axi = self.model_axi(x_axi)
        return self.classifier(torch.cat((x_sagt2, x_axi), dim=1))


class SpinalROIModelV3(nn.Module):
    def __init__(
        self,
        base_model,
        num_classes,
        in_channels=None,
        pretrained=True,
        rnn_hidden_size=512,
        rnn_num_layers=2,
        rnn_dropout=0,
        rnn_bidirectional=False,
    ):
        super().__init__()
        self.model_sagt2 = SplitROIFeatures(
            base_model,
            num_classes,
            in_channels=in_channels[0],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_axi = SplitROIFeatures(
            base_model,
            num_classes,
            in_channels=in_channels[1],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.classifier = nn.Linear(2 * rnn_hidden_size + 5, num_classes)

    def forward(self, x_sagt2, x_axi, level):
        x_sagt2 = self.model_sagt2(x_sagt2)
        x_axi = self.model_axi(x_axi)
        return self.classifier(torch.cat((x_sagt2, x_axi, level), dim=1))


class ForaminalROIModel(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.model_sagt1 = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x_sagt1, level, side):
        return self.model_sagt1(x_sagt1)


class ForaminalROIModelV2(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.model_sagt1 = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        self.model_axi = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        self.global_pool_sagt1 = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool_axi = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512 + 512, num_classes)

    def forward(self, x_sagt1, x_axi, level, side):
        x_sagt1 = self.model_sagt1.forward_features(x_sagt1)
        x_sagt1 = self.global_pool_sagt1(x_sagt1).flatten(start_dim=1)

        x_axi = self.model_sagt1.forward_features(x_axi)
        x_axi = self.global_pool_axi(x_axi).flatten(start_dim=1)

        x = torch.cat((x_sagt1, x_axi), dim=1)
        x = self.classifier(x)

        return x


class ForaminalROIModelV3(nn.Module):

    def __init__(
        self,
        base_model,
        num_classes,
        in_channels=None,
        pretrained=True,
        rnn_hidden_size=512,
        rnn_num_layers=2,
        rnn_dropout=0,
        rnn_bidirectional=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            in_chans=self.in_channels,
        )
        self.feature_num = self.model.num_features
        self.global_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.GRU(
            input_size=self.feature_num,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )
        if rnn_bidirectional:
            fc_in_size = rnn_hidden_size * 2
        else:
            fc_in_size = rnn_hidden_size
        self.classifier = nn.Linear(fc_in_size, num_classes)

    def forward(self, x, level, side):
        # Extract 3-channel subsets
        x_subset_list = []
        start_size = self.in_channels // 2
        end_size = self.in_channels - start_size
        for i in range(start_size, x.shape[1] - end_size + 1):
            x_subset = x[:, i - start_size : i + end_size, :, :]
            x_subset = self.model.forward_features(x_subset)
            x_subset = self.global_pool_2d(x_subset).flatten(start_dim=1)
            x_subset_list.append(x_subset)
        x_split = torch.stack(x_subset_list, dim=1)

        # RNN
        x_split, _ = self.rnn(x_split)
        x_split = x_split.permute(0, 2, 1)
        x_split = self.global_pool_1d(x_split).squeeze()

        return self.classifier(x_split)
    

# from monai.networks.nets.resnet import resnet18
# class ForaminalROIModelV4(nn.Module):

#     def __init__(
#         self,
#         base_model,
#         num_classes,
#         in_channels=None,
#         pretrained=True,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.model = timm.create_model(
#             model_name=base_model,
#             pretrained=pretrained,
#             in_chans=self.in_channels,
#         )
#         self.model = resnet18(pretrained=True, n_input_channels=1, feed_forward=False, shortcut_type='A')
#         self.classifier = nn.Linear(512, num_classes)

#     def forward(self, x, level, side):
#         x = x.unsqueeze(1)
#         x = self.model(x)
#         return self.classifier(x)


class SubarticularROIModel(nn.Module):
    def __init__(self, base_model, num_classes, in_channels=None, pretrained=True):
        super().__init__()
        self.model_sagt2 = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x_sagt2, level, side):
        x_sagt2 = self.model_sagt2(x_sagt2)
        return x_sagt2


class SubarticularROIModelV2(nn.Module):

    def __init__(
        self,
        base_model,
        num_classes,
        in_channels=None,
        pretrained=True,
        rnn_hidden_size=512,
        rnn_num_layers=2,
        rnn_dropout=0,
        rnn_bidirectional=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model = timm.create_model(
            model_name=base_model,
            pretrained=pretrained,
            in_chans=self.in_channels,
        )
        self.feature_num = self.model.num_features
        self.global_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.GRU(
            input_size=self.feature_num,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )
        if rnn_bidirectional:
            fc_in_size = rnn_hidden_size * 2
        else:
            fc_in_size = rnn_hidden_size
        self.classifier = nn.Linear(fc_in_size, num_classes)

    def forward(self, x, level, side):
        # Extract 3-channel subsets
        x_subset_list = []
        start_size = self.in_channels // 2
        end_size = self.in_channels - start_size
        for i in range(start_size, x.shape[1] - end_size + 1):
            x_subset = x[:, i - start_size : i + end_size, :, :]
            x_subset = self.model.forward_features(x_subset)
            x_subset = self.global_pool_2d(x_subset).flatten(start_dim=1)
            x_subset_list.append(x_subset)
        x_split = torch.stack(x_subset_list, dim=1)

        # RNN
        x_split, _ = self.rnn(x_split)
        x_split = x_split.permute(0, 2, 1)
        x_split = self.global_pool_1d(x_split).squeeze()

        if x_split.dim() == 1:
            x_split = x_split.unsqueeze(0)

        return self.classifier(x_split)


class SubarticularROIModelV3(nn.Module):
    def __init__(
        self,
        base_model,
        num_classes,
        in_channels=None,
        pretrained=True,
        rnn_hidden_size=512,
        rnn_num_layers=2,
        rnn_dropout=0,
        rnn_bidirectional=False,
    ):
        # TODO: future remove
        if isinstance(in_channels, int):
            in_channels = [1, 1]
        
        super().__init__()
        self.model_sagt2 = SplitROIFeatures(
            base_model,
            num_classes,
            in_channels=in_channels[0],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_axi = SplitROIFeatures(
            base_model,
            num_classes,
            in_channels=in_channels[1],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.classifier = nn.Linear(2 * rnn_hidden_size, num_classes)

    def forward(self, x_axi, x_sagt2, level, side):
        x_axi = self.model_axi(x_axi)
        x_sagt2 = self.model_sagt2(x_sagt2)
        return self.classifier(torch.cat((x_axi, x_sagt2), dim=1))
