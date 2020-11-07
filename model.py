import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34




class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = resnet18(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        # self.model.conv1 = nn.Conv2d(5, 64, kernel_size=3) # 인풋 레이어 5채널로 수정
        self.model.fc = nn.Linear(self.num_ftrs, 2) # 마지막 레이어 아웃풋 2로 수정(정상, 불량)
        print(self.model)

    def forward(self, x):
        # x = self.model.conv1
        # x = x.view(-1, 5, 224, 224)
        # x = F.relu(self.model.conv1(x))
        x = self.model(x)
        # x = F.relu(self.model(x))
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.model.fc(x))

        return x



class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()

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


class CRNN(nn.Module):
    class BidirectionalLSTM(nn.Module):

        def __init__(self, nIn, nHidden, nOut):
            super(self.BidirectionalLSTM, self).__init__()

            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
            self.embedding = nn.Linear(nHidden * 2, nOut)

        def forward(self, input):
            recurrent, _ = self.rnn(input)
            T, b, h = recurrent.size()
            t_rec = recurrent.view(T * b, h)

            output = self.embedding(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)

            return output

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            self.BidirectionalLSTM(512, nh, nh),
            self.BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        return output


def build_network(network_name):
    implemented_networks = ('resnet', 'VanillaCNN', 'UNet', 'CRNN',)
    assert network_name in implemented_networks, 'invaliad network name'
    network = {
        'resnet': Resnet(),
        'VanillaCNN': VanillaCNN(),
        # 'UNet': UNet(),
        # 'CRNN': CRNN(),
    }.get(network_name)
    return network
