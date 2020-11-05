import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34


class Peter_CNN(nn.Module):
    def __init__(self):
        super(Peter_CNN, self).__init__()

        self.model = resnet34(pretrained=True)        
        self.fc = nn.Linear(512, 2)
    
    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = F.relu(self.model(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))

        return F.log_softmax(x, dim=1)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s

        return num_features


def build_network(net_name):

    implemented_networks = ('Peter_CNN')

    assert net_name in implemented_networks

    net = None

    if net_name == 'Peter_CNN':
        net = Peter_CNN()
    
    return net