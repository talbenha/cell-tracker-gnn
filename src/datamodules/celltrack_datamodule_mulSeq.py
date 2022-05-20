import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from torch_geometric.data import DataLoader
from src.datamodules.datasets.graph_dataset import CellTrackDataset


def my_split(dataset, lengths, seq_len, sampler_type):
    """
    This function take as input a dataset and produce splitting to train and validation subset
    The splitting operation type is determined by "sampler_type" variable
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    if sampler_type == 'from_both':
        print(f"sample using {sampler_type}")
        min_max = lengths[1] // 2
        val_rand = seq_len[1] - min_max
        # print(f"split according to the first sequence (len={seq_len[1]}) minus {min_max} "
        #       f"meaning we do cyclic shift but remain in the end with half "
        #       f"of the len for validation ({min_max}) from each sequence")
        indices = np.roll(np.arange(sum(lengths)), val_rand)

    elif sampler_type == 'end_first':
        print(f"sample using {sampler_type}")
        indices = np.roll(np.arange(sum(lengths)), seq_len[1])

    elif sampler_type == 'end_sec':
        print(f"sample using {sampler_type}")
        indices = np.arange(sum(lengths))

    return [Subset(dataset, indices[(offset - length): offset]) for offset, length in zip(np.cumsum(lengths), lengths)]


class CellTrackDataModule(LightningDataModule):
    """
    LightningDataModule for cell graph dataset.

    This dataset is used to produce the final model to the cell tracking challenge -
    it can combine multiple sequences [often 2 sequence used for CTC]  in the training and validation sets
    determine by the 'train_val_test_split'

    A DataModule standardizes the train, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(
        self,
        dataset_params,
        sampler_type='from_both',
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: list = [80, 20, 0],

        **kwargs,
    ):
        super().__init__()

        self.sampler_type = sampler_type
        self.train_val_test_split = train_val_test_split
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = None
        self.dims = (1, 16)

        self.data: CellTrackDataset = CellTrackDataset(**dataset_params, split='train')
        self.seq_len = self.data.train_seq_len_check

        print(f"seq len : {self.data.train_seq_len_check}")
        self.data_train: CellTrackDataset = None
        self.data_val: CellTrackDataset = None
        self.data_test: CellTrackDataset = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = self.data
        data_len = len(self.data)
        train_val_test_split = self.train_val_test_split
        train_val_test_split = np.array(train_val_test_split)
        train_val_test_split = data_len * train_val_test_split / train_val_test_split.sum()
        self.train_val_test_split = train_val_test_split.astype('int32')
        self.train_val_test_split[0] += data_len - self.train_val_test_split.sum()

        self.data_train, self.data_val, self.data_test = my_split(
            dataset, self.train_val_test_split, self.seq_len, self.sampler_type
        )

        print(f"Training dataset length: {len(self.data_train)}")
        print(f"Validation dataset length: {len(self.data_val)}")
        print(f"Test dataset length: {len(self.data_test)}")
        self.data_test = self.data_val

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
