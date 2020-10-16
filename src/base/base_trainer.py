from abc import ABC, abstractmethod
from .base_dataset import BaseADDataset, BaseCNNDataset
from .base_net import BaseNet


class BaseTrainer(ABC):

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, batch_size: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    @abstractmethod
    def train(self, dataset, net: BaseNet) -> BaseNet:
        pass

    
