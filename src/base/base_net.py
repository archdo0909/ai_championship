import logging
import torch.nn as nn
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural network"""

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def summary(self, input_shape):
        """Network summary"""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)