from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseADDataset(ABC):

    def __init__(self, root: str):
        super().__init__()
        self.root = root # root path to data

        self.n_classes = 2
        self.train_set = None
        self.test_set = None

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True,
                shuffle_test=False, num_worers: int=0) -> (DataLoader, DataLoader):
                pass
    
    def __repr__(self):
        return self.__class__.__name__

class BaseCNNDataset(ABC):
    """CNN image classifier dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__