from torch.utils.data import DataLoader

from .datasets import RSNA2024Dataset


class RSNADataLoader(DataLoader):
    def __init__(
        self, df, transforms, phase, data_dir, out_vars, batch_size, shuffle, num_workers, pin_memory
    ):
        dataset = RSNA2024Dataset(df, data_dir, out_vars, transforms)
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
