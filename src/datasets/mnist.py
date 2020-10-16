from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset

import torch
import torchvision.transforms as transform
import random

class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root):
        super().__init__(root)
        
        transform= transforms.ToTensor()

        self.train_set = MyMNIST(root=self.root, train=True, transform=transform, target_tranform=target_transform,
                            download=True)

        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                download=True)

class MyMNIST(MNIST):

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):

        img, target = self.data[index], ing(self.targets[index])

        # 8-bit pixels, black and white
        img = Image.fromarray(img.numpy(), mode='L
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index