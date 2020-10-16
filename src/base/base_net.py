import torch.nn as nn
import torchsummary.summary as summary
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural network"""

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def summary(self, input_shape):
        summary(self.model, input_shape)