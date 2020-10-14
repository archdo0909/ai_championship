from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseADDataset(ABC):

    def __init__(self):
        super().__init__()
        

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True,
                shuffle_test=False, num_worers: int=0) -> (DataLoader, DataLoader):
                pass
    
    def __repr__(self):
        return self.__class__.__name__
