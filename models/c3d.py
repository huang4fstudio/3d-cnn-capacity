import torch.nn as nn
import torch
from torch import flatten
from torch import cat 
from typing import Union, Dict

from .conv3d_bn import Conv3d_BN        

class C3D(nn.Module):

    def __init__(self, in_channels, out_channels, pretrained_net=None, num_params_factor=1.0):
        super(C3D, self).__init__()
        self.num_params_factor = num_params_factor
        self.conv_1a = Conv3d_BN(in_channels, int(64 * num_params_factor), (3, 3, 3), padding=1, bn=False)
      
        self.maxpool_1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)) # 64 x 112 x 112
        self.conv_2a = Conv3d_BN(int(64 * num_params_factor), int(128 * num_params_factor), (3, 3, 3), padding=1, bn=False)
        
        self.maxpool_2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)) # 32 x 56 x 56

        self.conv_3a = Conv3d_BN(int(128 * num_params_factor), int(256 * num_params_factor), (3, 3, 3), padding=1)
        self.conv_3b = Conv3d_BN(int(256 * num_params_factor), int(256 * num_params_factor), (3, 3, 3), padding=1)

        self.maxpool_3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)) # 16 x 28 x 28

        self.conv_4a = Conv3d_BN(int(256 * num_params_factor), int(512 * num_params_factor), (3, 3, 3), padding=1)
        self.conv_4b = Conv3d_BN(int(512 * num_params_factor), int(512 * num_params_factor), (3, 3, 3), padding=1)

        self.maxpool_4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)) # 8 x 14 x 14

        self.conv_5a = Conv3d_BN(int(512 * num_params_factor), int(512 * num_params_factor), (3, 3, 3), padding=1)
        self.conv_5b = Conv3d_BN(int(512 * num_params_factor), int(512 * num_params_factor), (3, 3, 3), padding=1)

        self.maxpool_5 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)) # 4 x 7 x 7

        self.fc6 = Conv3d_BN(int(4096 * num_params_factor), int(4096 * num_params_factor), (4, 7, 7))
        self.fc7 = nn.Conv3d(int(4096 * num_params_factor), out_channels, (1, 1, 1))
    
    
    def forward(self, x):
        c1a_out = self.conv_1a(x)
        m1_out = self.maxpool_1(c1a_out)
        
        c2a_out = self.conv_2a(m1_out)
        m2_out = self.maxpool_2(c2a_out)

        c3a_out = self.conv_3a(m2_out)
        c3b_out = self.conv_3b(c3a_out)
        
        m3_out = self.maxpool_3(c3b_out)
        
        c4a_out = self.conv_4a(m3_out)
        c4b_out = self.conv_4b(c4a_out)

        m4_out = self.maxpool_4(c4b_out)

        c5a_out = self.conv_5a(m4_out)
        c5b_out = self.conv_5b(c5a_out)
        
        m5_out = self.maxpool_5(c5b_out)
        f6_out = self.fc6(m5_out)
        f_out = self.fc7(f6_out)
        out_logits = torch.squeeze(f_out)
        #out_logits = flatten(f_out)
        return out_logits
