import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class LG_LeNet(BaseNet):

    def __init__(self, rep_dim=32):
        