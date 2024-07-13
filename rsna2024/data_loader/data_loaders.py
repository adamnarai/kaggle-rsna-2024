from torch.utils.data import DataLoader

from .datasets import (
    RSNA2024Dataset,
    RSNA2024SplitDataset,
    RSNA2024SplitCoordDataset,
    RSNA2024SplitKpmapDataset,
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
    ):
        dataset_instance = dataset(df, data_dir, out_vars, img_num=img_num, transform=transform)
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
        super().__init__(RSNA2024Dataset, *args, **kwargs)


class RSNASplitDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNA2024SplitDataset, *args, **kwargs)


class RSNASplitCoordDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNA2024SplitCoordDataset, *args, **kwargs)



class RSNASplitKpmapDataLoader(BaseRSNADataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(RSNA2024SplitKpmapDataset, *args, **kwargs)