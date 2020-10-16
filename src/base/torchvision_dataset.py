from base.base_dataset import BaseCNNDataset
from torch.utils.data import DataLoader

class TorchvisionDataset(BaseCNNDataset):

    def __init__(self, root: str):
        super().__init__(root)
    
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int=0) -> (
        DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader