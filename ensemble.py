import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from model import Resnet
from model import CRNN
from model import UNet

class EnsembleNetwork(nn.Module):
    def __init__(self):
        super(EnsembleNetwork, self).__init__()

        # Init dual class classifier
        self.resnet = Resnet()
        self.crnn = CRNN()
        self.unet = UNet()

        # Init one class classifier
        self.deep_sad_normal = None
        self.deep_sad_abnormal = None

        # Init models list
        self.models = [self.resnet, self.crnn, self.unet]

        # Load weights for non-anomaly detectors
        self.resnet.load_state_dict(torch.load('/workspace/demon/resnet_random700_spectrogram.pt'))
        self.crnn.load_state_dict(torch.load('/workspace/demon/crnn_random700_spectrogram.pt'))
        self.unet.load_state_dict(torch.load('/workspace/demon/unet_random700_spectrogram.pt'))

        # Load weights for anomaly detectors
        #self.deep_sad_normal.load(torch.load())
        #self.deep_sad_abnormal.load(torch.load())

        # Freeze parameter values
        for model in self.models:
            for param in model.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        # make prediction for deep sads
        result_sad_normal = 0  #self.deep_sad_normal.forward(x)
        result_sad_abnormal = 1  #self.deep_sad_abnormal.forward(x)
        if result_sad_normal == result_sad_abnormal:
            return result_sad_abnormal
        
        # if not in a consensus, try non-anomaly detectors
        result_resnet = self.resnet.forward(x)
        result_crnn = self.crnn.forward(x)
        result_unet = self.unet.forward(x)
        
        return (result_resnet + result_crnn) / 3


if __name__ == '__main__':
    # Create ensemble model
    model = EnsembleNetwork()

    # TODO: Load an actual input
    x = torch.randn(1, 5, 100, 100)

    # Make prediction
    output = model(x)
    print(output)
