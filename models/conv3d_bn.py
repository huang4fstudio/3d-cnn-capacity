import torch.nn as nn
from typing import Union

class Conv3d_BN(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 kernel : Union[list, tuple],
                 stride: Union[int, list, tuple] = 1,
                 padding: int = 0,
                 bn : bool = True,
                 activation : nn.Module = nn.ReLU()):
        super(Conv3d_BN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
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