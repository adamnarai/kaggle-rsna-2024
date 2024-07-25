from torch.utils.data import DataLoader
import torch

from .datasets import (
    RSNADataset,
    RSNAMeanposDataset,
    RSNASplitDataset,
    RSNASplitCoordDataset,
    RSNAMilSplitDataset,
    RSNASplitMeanposDataset,
)


class BaseRSNADataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        df,
        transform,
        phase,
        data_dir,
        out_vars,
        img_num,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
        resolution,
        block_position=None,
        series_mask=None,
    ):
        dataset_instance = dataset(df, data_dir, out_vars, img_num=img_num, transform=transform, resolution=resolution, block_position=block_position, series_mask=series_mask)
        if phase == 'valid':
            shuffle = False
        super().__init__(
            dataset_instance,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class RSNADataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNADataset, *args, **kwargs)


class RSNAMeanposDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNAMeanposDataset, *args, **kwargs)


class RSNASplitMeanposDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNASplitMeanposDataset, *args, **kwargs)


class RSNASplitDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNASplitDataset, *args, **kwargs)


class RSNASplitCoordDataLoader(DataLoader):
    def __init__(
        self,
        df,
        transform,
        phase,
        data_dir,
        out_vars,
        img_num,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
        resolution,
        heatmap_std,
    ):
        dataset_instance = RSNASplitCoordDataset(
            df=df,
            data_dir=data_dir,
            out_vars=out_vars,
            img_num=img_num,
            transform=transform,
            resolution=resolution,
            heatmap_std=heatmap_std,
        )
        if phase == 'valid':
            shuffle = False
        super().__init__(
            dataset_instance,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

class RSNAMilSplitDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNAMilSplitDataset, *args, **kwargs)
