import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_channels, out_channels):
    unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
            )
    return unit

def deconv_bn_relu(in_channels, out_channels, drop_out=False):
    unit = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(),
            nn.ReLU()
            )
    return unit

class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pix2PixGenerator, self).__init__()
        self.convs = [
        nn.Conv2d(in_channels, 64, 3, 1, 1),
        conv_bn_relu(64, 128),
        conv_bn_relu(128, 256),
        conv_bn_relu(256, 512),
        conv_bn_relu(512, 512),
        conv_bn_relu(512, 512),
        conv_bn_relu(512, 512),
        conv_bn_relu(512, 512)
        ]

        self.deconvs = [
        deconv_bn_relu(512, 512, drop_out=True),
        deconv_bn_relu(1024, 512, drop_out=True),
        deconv_bn_relu(1024, 512, drop_out=True),
        deconv_bn_relu(1024, 512),
        deconv_bn_relu(1024, 256),
        deconv_bn_relu(512, 128),
        deconv_bn_relu(256, 64),
        nn.ConvTranspose2d(64, out_channels, 3, 1, 1)
        ]

    def forward(self, x):
        hs = []
        h = self.convs[0](x)
        for conv in self.convs[1: -1]:
            h = conv(h)
            hs.append(h)
        h = self.convs[-1](h)

        h = self.deconvs[0](h)
        for i, deconv in enumerate(self.deconvs[1: -1]):
            h = torch.cat((h, hs[-(i+1)]), dim=1)
            h = deconv(h)
        h = self.deconvs[-1](h)

        return h

class Pix2PixDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pix2PixDiscriminator, self).__init__()
        self.conv1_0 = conv_bn_relu(in_channels, 32)
        self.conv1_1 = conv_bn_relu(out_channels, 32)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = nn.Conv2d(512, 1, 3, 1, 1)

    def __call__(self, x_0, x_1):
        h = self.conv2(torch.cat((self.conv1_0(x_0), self.conv1_1(x_1)), dim=1))
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = F.adaptive_avg_pool2d(h, 1)
        return h.view(-1, 1)
