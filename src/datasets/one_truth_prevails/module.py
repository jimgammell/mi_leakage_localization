

from .dataset import OneTruthPrevails

class DataModule(L.LightningDataModule):
    def __init__(self,
        root: str,
        train_batch_size=1024,
        eval_batch_size=10240,
        dataset_kwargs={},
        dataloader_kwargs={},
        train_prop=0.1,
        val_prop=0.01
    ):
        self.root = root
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.train_prop = train_prop
        self.val_prop = val_prop
        super().__init__()