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


class LG_LeNet():

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(4, 3, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(3, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(3 * 27 * 27, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class LG_LeNet_Decoder():

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim

        # Decoder network
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class LG_LeNet_Autoencoder():

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = LG_LeNet(rep_dim=rep_dim)
        self.decoder = LG_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def build_network(net_name):

    implemented_networks = ('Peter_CNN', 'LG_LeNet', 'LG_LeNet_Autoencoder')

    assert net_name in implemented_networks

    net = None

    if net_name == 'Peter_CNN':
        net = Peter_CNN()

    if net_name == 'LG_LeNet':
        net = LG_LeNet()

    if net_name == 'LG_LeNet_Autoencoder':
        net = LG_LeNet_Autoencoder()
    
    return net