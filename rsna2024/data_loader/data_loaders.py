from torch.utils.data import DataLoader

from .datasets import (
    RSNADataset,
    RSNADatasetMeanpos,
    RSNASplitDataset,
    RSNASplitCoordDataset,
    RSNASplitKpmapDataset,
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
        block_position,
        series_mask,
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


class RSNADatasetMeanposDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNADatasetMeanpos, *args, **kwargs)


class RSNASplitMeanposDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNASplitMeanposDataset, *args, **kwargs)


class RSNASplitDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNASplitDataset, *args, **kwargs)


class RSNASplitCoordDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNASplitCoordDataset, *args, **kwargs)


class RSNASplitKpmapDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNASplitKpmapDataset, *args, **kwargs)


class RSNAMilSplitDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNAMilSplitDataset, *args, **kwargs)