import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34


class PeterCNN(nn.Module):
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


class UNet(nn.Module):
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.05),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.05),
        )   

    def __init__(self, in_channels, out_channels):
        super().__init__()
                
        self.dconv_down1 = self.double_conv(in_channels, 64)
        self.dconv_down2 = self.double_conv(64, 128)
        self.dconv_down3 = self.double_conv(128, 256)
        self.dconv_down4 = self.double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_up3 = self.double_conv(256 + 512, 256)
        self.dconv_up2 = self.double_conv(128 + 256, 128)
        self.dconv_up1 = self.double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        return self.conv_last(x)

def build_network(network_name):
    implemented_networks = ('PeterCNN', 'UNet',)
    assert network_name in implemented_networks, 'invaliad network name'
    network = {
        'vanilla_cnn': PeterCNN(),
        'UNet': UNet(),
    }.get(network_name)
    return network

