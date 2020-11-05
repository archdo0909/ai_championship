import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class EnsembleNetwork(nn.Module):
    def __init__(self, model_1, model_2, model_3, nb_classes=2):
        super(EnsembleNetwork, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3

        # Now Remove the Last layer
        self.model_1 = nn.Identity()
        self.model_2 = nn.Identity()
        self.model_3 = nn.Identity()

        # Now we a head on top of those models
        # The trained model I am passing here is binary type based on densenet169
        # so I added a sequential `nn.Linear(num_ftrs, 2)` with them for output when they were trained
        # so, sould I use 2 instead of 1664 ? but used 1664 any way.
        self.classifier = nn.Linear(
            1664 + 1664 + 1664 + 1664 + 1664 + 1664 + 1664, nb_classes
        )

    def forward(self, x):
        # clone to make sure x is not changed by inplace methods
        x1 = self.model_1(x.clone())
        x1 = x1.view(x1.size(0), -1)

        x2 = self.model_2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.model_3(x)
        x3 = x3.view(x3.size(0), -1)

        # final
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(F.relu(x))
        return x


if __name__ == '__main__':
    # We use pretrained torchvision models here
    modelA = models.resnet50(pretrained=True)
    modelB = models.resnet18(pretrained=True)
    modelC = models.resnet18(pretrained=True)

    # Freeze these models
    for param in modelA.parameters():
        param.requires_grad_(False)

    for param in modelB.parameters():
        param.requires_grad_(False)

    for param in modelC.parameters():
        param.requires_grad_(False)

    # Create ensemble model
    model = EnsembleNetwork(modelA, modelB, modelC)

    # Load input
    x = torch.randn(1, 3, 224, 224)

    # Make prediction
    output = model(x)
