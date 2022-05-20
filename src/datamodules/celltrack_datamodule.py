from typing import Optional
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from src.datamodules.datasets.graph_dataset import CellTrackDataset

class CellTrackDataModule(LightningDataModule):
    """

    LightningDataModule for cell graph dataset.

    A DataModule standardizes the train, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(
        self,
        dataset_params,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = None
        self.dims = (1, 16)

        self.data_train: CellTrackDataset = CellTrackDataset(**dataset_params, split='train')
        self.data_val: CellTrackDataset = CellTrackDataset(**dataset_params, split='valid')
        self.data_test: CellTrackDataset = CellTrackDataset(**dataset_params, split='test')

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        pass

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
