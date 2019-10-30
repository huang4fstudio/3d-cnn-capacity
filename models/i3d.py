import torch.nn as nn
from torch import cat 


class Conv3d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, bn=True, activation=nn.ReLU):
        super(Conv3d_BN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (1, 1, 1))
        self.activation = activation
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm3d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
        
 
class InceptionBlock(nn.Module):
    def __init__(self):
        super(InceptionBlock, self).__init__()
        # branches counted from left to right, from p.5: https://arxiv.org/pdf/1705.07750.pdf
        self.branch1 = Conv3d_BN(, , (1, 1, 1))
        
        self.branch2 = nn.Sequential(Conv3d_BN(, , (1, 1, 1)), 
            Conv3d_BN(, , (3, 3, 3)))
        )

        self.branch3 = nn.Sequential(Conv3d_BN(, , (1, 1, 1)), 
            Conv3d_BN(, , (3, 3, 3)))
        )

        self.branch4 = nn.Sequential(nn.MaxPool3d((3, 3, 3)), 
            Conv3d_BN(, , (1, 1, 1)))
        )
    
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return cat([x1, x2, x3, x4])


class I3D(nn.Module):
    def __init__(self):
        super(I3D, self).__init__()
