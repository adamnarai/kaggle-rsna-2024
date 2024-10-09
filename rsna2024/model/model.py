import logging
import timm
import segmentation_models_pytorch as smp

import torch
from torch import nn

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
            num_classes=0,
        )
        self.feature_num = self.model.num_features
        self.global_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.GRU(
            input_size=self.feature_num,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        x_subset_list = []
        start_size = self.in_channels // 2
        end_size = self.in_channels - start_size
        for i in range(start_size, x.shape[1] - end_size + 1):
            x_subset = x[:, i - start_size : i + end_size, :, :]
            x_subset = self.model(x_subset)
            x_subset_list.append(x_subset)
        x_split = torch.stack(x_subset_list, dim=1)

        # RNN
        x_split, _ = self.rnn(x_split)
        x_split = x_split.permute(0, 2, 1)
        x_split = self.global_pool_1d(x_split).squeeze()
        if x_split.dim() == 1:
            x_split = x_split.unsqueeze(0)

        return x_split


class SpinalROIModel(nn.Module):
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
            in_channels=in_channels[0],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_axi = SplitROIFeatures(
            base_model,
            in_channels=in_channels[1],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        if rnn_bidirectional:
            fc_in_size = rnn_hidden_size * 2
        else:
            fc_in_size = rnn_hidden_size
        self.classifier = nn.Linear(2 * fc_in_size, num_classes)

    def forward(self, x_sagt2, x_axi, level):
        x_sagt2 = self.model_sagt2(x_sagt2)
        x_axi = self.model_axi(x_axi)
        return self.classifier(torch.cat((x_sagt2, x_axi), dim=1))


class ForaminalROIModel(nn.Module):
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
        self.model = SplitROIFeatures(
            base_model,
            in_channels=in_channels,
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        if rnn_bidirectional:
            fc_in_size = rnn_hidden_size * 2
        else:
            fc_in_size = rnn_hidden_size
        self.classifier = nn.Linear(fc_in_size, num_classes)

    def forward(self, x, level, side):
        x = self.model(x)
        return self.classifier(x)


class SubarticularROIModel(ForaminalROIModel):
    pass


class GlobalROIModel(nn.Module):
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
            in_channels=in_channels[0],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_sagt1 = SplitROIFeatures(
            base_model,
            in_channels=in_channels[1],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_axi = SplitROIFeatures(
            base_model,
            in_channels=in_channels[2],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        if rnn_bidirectional:
            fc_in_size = rnn_hidden_size * 2
        else:
            fc_in_size = rnn_hidden_size
        self.classifier = nn.Linear(5 * fc_in_size, num_classes)

    def forward(self, x_sagt2, x_sagt1_left, x_sagt1_right, x_axi_left, x_axi_right, level):
        x_sagt2 = self.model_sagt2(x_sagt2)
        x_sagt1_left = self.model_sagt1(x_sagt1_left)
        x_sagt1_right = self.model_sagt1(x_sagt1_right)
        x_axi_left = self.model_axi(x_axi_left)
        x_axi_right = self.model_axi(x_axi_right)
        features = torch.cat((x_sagt2, x_sagt1_left, x_sagt1_right, x_axi_left, x_axi_right), dim=1)
        return self.classifier(features)


class StudyROIModel(nn.Module):
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
            in_channels=in_channels[0],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_sagt1 = SplitROIFeatures(
            base_model,
            in_channels=in_channels[1],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        self.model_axi = SplitROIFeatures(
            base_model,
            in_channels=in_channels[2],
            pretrained=pretrained,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            rnn_bidirectional=rnn_bidirectional,
        )
        if rnn_bidirectional:
            self.embedding_size = rnn_hidden_size * 2
        else:
            self.embedding_size = rnn_hidden_size

        self.positional_embedder = nn.Linear(5, self.embedding_size)
        self.sequence_embedder = nn.Linear(5, self.embedding_size)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=8,
            dim_feedforward=self.embedding_size,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=2)
        self.classifier = nn.Linear(self.embedding_size, 3)

    def forward(self, x_sagt2, x_sagt1_left, x_sagt1_right, x_axi_left, x_axi_right):
        x_sagt2 = self.model_sagt2(x_sagt2.flatten(0, 1)).unflatten(0, [-1, 5])
        x_sagt1_left = self.model_sagt1(x_sagt1_left.flatten(0, 1)).unflatten(0, [-1, 5])
        x_sagt1_right = self.model_sagt1(x_sagt1_right.flatten(0, 1)).unflatten(0, [-1, 5])
        x_axi_left = self.model_axi(x_axi_left.flatten(0, 1)).unflatten(0, [-1, 5])
        x_axi_right = self.model_axi(x_axi_right.flatten(0, 1)).unflatten(0, [-1, 5])
        x = torch.stack([x_sagt2, x_sagt1_left, x_sagt1_right, x_axi_left, x_axi_right], dim=1)
        bs, _, _, _ = x.shape

        one_hot_position_encoding = torch.eye(5).repeat(bs, 5, 1, 1).to(x.device)
        one_hot_sequence_encoding = torch.eye(5).repeat(bs, 5, 1, 1).swapaxes(1, 2).to(x.device)

        positional_embedding = self.positional_embedder(one_hot_position_encoding)
        sequence_embedding = self.sequence_embedder(one_hot_sequence_encoding)

        input_tokens = x + positional_embedding + sequence_embedding
        input_tokens = input_tokens.flatten(1, 2)

        output_tokens = self.transformer_encoder(input_tokens)
        preds = self.classifier(output_tokens).swapaxes(1, 2).flatten(1, 2)

        return preds
