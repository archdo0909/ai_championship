from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.lg_dataset import LGDataset

import torch

class LGADDataset(BaseADDataset):

    def __init__(self, root: str, dataset_name, random_state=None):
        super().__init__(root)

        # Define normal and outlier classes 
        self.n_classes = 2 # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        self.train_set = LGDataset(root=self.root, dataset_name=dataset_name,
                                   train=True, random_state=random_state)

        self.test_set = LGDataset(root=self.root, dataset_name=dataset_name,
                                  train=True, random_state=random_state)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int=0) -> (
            DataLoader, DataLoader):
        
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        
        return train_loader, test_loader