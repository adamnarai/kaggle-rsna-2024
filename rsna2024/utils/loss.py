import torch
from torch.nn import CrossEntropyLoss


class RSNACrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, weight=torch.tensor([1.0, 2.0, 4.0]), reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, input, target):
        return super().forward(torch.unflatten(input, 1, [3, -1]), target)
