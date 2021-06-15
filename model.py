import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """U-net paper http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/"""
    def __init__(self, num_layers):
        super(UNet, self).__init__()
        
        self.convin = AddDoubleConv(1, 64)
        self.downsample1 = Encode(64, 128) 
        self.downsample2 = Encode(128, 256)
        self.downsample3 = Encode(256, 512)
        self.downsample4 = Encode(512, 1024)
        
        self.upsample1 = Decode(1024, 512)
        self.upsample2 = Decode(512, 256)
        self.upsample3 = Decode(256, 128)
        self.upsample4 = Decode(128, 64)
        self.convout = ConvOut(64, num_layers)

    def forward(self, x):
        x1 = self.convin(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)
        x = self.upsample1(x5, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)
        return self.convout(x)

class AddDoubleConv(nn.Module):
    """Increase Accuracy by Batch normalization"""

    def __init__(self, in_channels, out_channels, mid_channels=None, addBN=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            if addBN:
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                else:
                    self.double_conv = nn.Sequential(
                        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    )


    def forward(self, x):
        return self.double_conv(x)


    
class Encode(nn.Module):
    """Halves the number of feature channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            AddDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mp_conv(x)


class Decode(nn.Module):
    """"Double the number of feature channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = AddDoubleConv(in_channels, out_channels // 2, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.decode(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvOut(nn.Module):
    """Map 64-component feature vector to the desired layers"""
    def __init__(self, in_channels, out_channels):
        super(ConvOut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
