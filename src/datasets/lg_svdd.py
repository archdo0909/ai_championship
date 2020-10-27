from torch.utils.data import DataLoader, Subset
from torch.utils.data import Subset
from .preprocessing import get_target_label_idx
from base.base_dataset import BaseADDataset
from base.lg_dataset import LGDataset

import torchvision.transforms as transforms


class LG_SVDD_Dataset(BaseADDataset):

    def __init__(self, dataset_name: str, root: str, normal_class=5, random_state: int = 0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = tuple([1])
        self.dataset_name = dataset_name

        train_set = LGDataset(root=self.root, dataset_name=dataset_name,
                              train=True, random_state=random_state)

        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = LGDataset(root=self.root, dataset_name=dataset_name,
                                  train=False, random_state=random_state)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        
        return train_loader, test_loader