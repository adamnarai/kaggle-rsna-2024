from torch.utils.data import DataLoader

from .datasets import RSNA2024Dataset, RSNA2024SplitDataset


class RSNADataLoader(DataLoader):
    def __init__(
        self, df, transform, phase, data_dir, out_vars, img_num, batch_size, shuffle, num_workers, pin_memory
    ):
        dataset = RSNA2024Dataset(df, data_dir, out_vars, img_num=img_num, transform=transform)
        if phase == 'train':
            pass
        elif phase == 'valid':
            shuffle = False
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

class RSNASplitDataLoader(DataLoader):
    def __init__(
        self, df, transform, phase, data_dir, out_vars, img_num, batch_size, shuffle, num_workers, pin_memory
    ):
        dataset = RSNA2024SplitDataset(df, data_dir, out_vars, img_num=img_num, transform=transform)
        if phase == 'train':
            pass
        elif phase == 'valid':
            shuffle = False
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
