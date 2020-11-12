import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from model import Resnet
from model import VanillaCNN


class EnsembleNetwork(nn.Module):
    def __init__(self, model_1, model_2):
        super(EnsembleNetwork, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2

        # classifier
        self.classifier = nn.Linear(4, 1)

    def forward(self, x):
        # clone to make sure x is not changed by inplace methods
        x1 = self.model_1(x.clone())
        x1 = x1.view(x1.size(0), -1)

        x2 = self.model_2(x)
        x2 = x2.view(x2.size(0), -1)

        # final
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(x)
        x = self.classifier(x)
        x = nn.Sigmoid()(x)
        return x


if __name__ == '__main__':
    # Load our supervised models here
    modelA = Resnet()
    modelB = VanillaCNN()

    modelA.load_state_dict(torch.load('/workspace/jinsung/y'))
    modelB.load_state_dict(torch.load('/workspace/jinsung/x'))

    # Freeze these models
    for param in modelA.parameters():
        param.requires_grad_(False)

    for param in modelB.parameters():
        param.requires_grad_(False)

    # Create ensemble model
    model = EnsembleNetwork(modelA, modelB)

    # TODO: Load an actual input
    x = torch.randn(1, 5, 100, 100)

    # Make prediction
    output = model(x)
    print(output)
