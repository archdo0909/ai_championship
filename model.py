import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.conv_in = nn.Conv2d(3, 16, 3, padding=1)
        self.model = resnet18(pretrained=True)

        self.model.conv1 = nn.Conv2d(5, 64, 3, padding=1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return F.sigmoid(self.model(x))


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()

        self.model = resnet34(pretrained=True)
        self.model.conv1 = nn.Conv2d(5, 64, 3, padding=1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        return F.sigmoid(self.model(x))


class UNet(nn.Module):
    def double_conv(self, in_channels, out_channels):
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

    def __init__(self, in_channels=5, out_channels=1):
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

        self.conv_real_last = nn.Linear(128 * 128, 2)
        
    def forward(self, x):
        # reshape mini-batched tensor when reshaping, take pixel's region account
        x = F.interpolate(x, size=(128, 128), mode='bicubic', align_corners=False)

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
        x = self.conv_last(x)
        
        x = torch.reshape(x, shape=(-1, 128*128))
        x = self.conv_real_last(x)
        return F.sigmoid(x)


class BidirectionalLSTM(nn.Module):

        def __init__(self, nIn, nHidden, nOut):
            super(BidirectionalLSTM, self).__init__()

            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
            self.embedding = nn.Linear(nHidden * 2, nOut)

        def forward(self, input):
            recurrent, _ = self.rnn(input)
            T, b, h = recurrent.size()
            t_rec = recurrent.view(T * b, h)

            output = self.embedding(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)

            return output

class CRNN(nn.Module):
    
    def __init__(self, imgH=112, nc=5, nclass=2, nh=3, n_rnn=2, leakyRelu=False):
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
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # reshape mini-batched tensor when reshaping, take pixel's region account
        input = F.interpolate(input, size=(128, 128), mode='bicubic', align_corners=False)
        
        # conv features
        conv = self.cnn(input) # 6 512 7 33 : conv.shape
        b, c, x, y = conv.shape
        #conv = F.interpolate(conv, size=(b, c, 1, y), mode='bicubic', align_corners=False)
        conv = torch.reshape(conv, shape=(b, c, 1, -1))

        b, c, h, w = conv.size()
        assert h == 1, f"h: {h} where the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        return output[0]


class LG_LeNet(nn.Module):

    def __init__(self, rep_dim=98):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(4, 3, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(3, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(3 * 28 * 28, self.rep_dim, bias=False)

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


class LG_LeNet_Decoder(nn.Module):

    def __init__(self, rep_dim=98):
        super().__init__()

        self.rep_dim = rep_dim

        # Decoder network
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 3, 5, bias=False, padding=2)

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / 49), 7, 7)
        x = F.interpolate(F.leaky_relu(x), scale_factor=4)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=4)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class LG_LeNet_Autoencoder(nn.Module):

    def __init__(self, rep_dim=98):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = LG_LeNet(rep_dim=rep_dim)
        self.decoder = LG_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def build_network(network_name):
    implemented_networks = ('resnet', 'VanillaCNN', 'UNet', 'CRNN', 'LG_LeNet', 'LG_LeNet_Autoencoder')
    assert network_name in implemented_networks, 'invaliad network name'
    network = {
        'resnet': Resnet(),
        'VanillaCNN': VanillaCNN(),
        'UNet': UNet(),
        'CRNN': CRNN(),
        'LG_LeNet': LG_LeNet(),
        'LG_LeNet_Autoencoder': LG_LeNet_Autoencoder(),
    }.get(network_name)
    return network
