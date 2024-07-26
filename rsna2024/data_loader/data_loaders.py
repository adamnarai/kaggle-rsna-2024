from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):

    def __init__(
        self,
        dataset,
        phase,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
    ):
        if phase == 'valid':
            shuffle = False
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
