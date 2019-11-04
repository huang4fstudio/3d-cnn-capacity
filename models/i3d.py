import torch.nn as nn
import torch
from torch import flatten
from torch import cat 
from typing import Union, Dict
#import torchvision.models as models

INCEPTION_BLOCK_FILTER_SPECS = {
    '3a': [64, 96, 128, 16, 32 ,32],
    '3b': [128, 128, 192, 32, 96, 64],
    '4a': [192, 96, 208, 16, 48, 64],
    '4b': [160, 112, 224, 24, 64, 64],
    '4c': [128, 128, 256, 24, 64, 64],
    '4d': [112, 144, 288, 32, 64, 64],
    '4e': [256, 160, 320, 32, 128, 128],
    '5a': [256, 160, 320, 32, 128, 128],
    '5b': [384, 192, 384, 48, 128, 128]
}

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
        

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, conv_out_channels : Dict[str, int]):
        super(InceptionBlock, self).__init__()
        # branches counted from left to right, from p.5: https://arxiv.org/pdf/1705.07750.pdf

        self.branch_1x1 = Conv3d_BN(in_channels, conv_out_channels['1x1'], (1, 1, 1), padding=0)
        
        self.branch_3x3_1 = nn.Sequential(Conv3d_BN(in_channels, conv_out_channels['3x3_reduce1'], (1, 1, 1)), 
            Conv3d_BN(conv_out_channels['3x3_reduce1'], conv_out_channels['3x3_1'], (3, 3, 3), padding=1)
        )

        self.branch_3x3_2 = nn.Sequential(Conv3d_BN(in_channels, conv_out_channels['3x3_reduce2'], (1, 1, 1)), 
            Conv3d_BN(conv_out_channels['3x3_reduce2'], conv_out_channels['3x3_2'], (3, 3, 3), padding=1)
        )

        self.branch_pool = nn.Sequential(nn.MaxPool3d((3, 3, 3), stride=1), 
            Conv3d_BN(in_channels, conv_out_channels['pool_proj'], (1, 1, 1), padding=1)
        )
    
    def forward(self, x):
        x1 = self.branch_1x1(x)
        x2 = self.branch_3x3_1(x)
        x3 = self.branch_3x3_2(x)
        x4 = self.branch_pool(x)
        return cat([x1, x2, x3, x4], dim=1)


class I3D(nn.Module):

    def __init__(self, in_channels, out_channels, pretrained_net=None):
        super(I3D, self).__init__()
        self.conv_1 = Conv3d_BN(in_channels, 64, (7, 7, 7), stride=2, padding=3)

        self.maxpool_1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv_2 = Conv3d_BN(64, 64, (1, 1, 1))
        self.conv_3 = Conv3d_BN(64, 192, (3, 3, 3), padding=1)

        self.maxpool_2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.inception_3a = InceptionBlock(192, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['3a']))
        self.inception_3b = InceptionBlock(256, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['3b']))

        self.maxpool_3 = nn.MaxPool3d((3, 3, 3), stride=2, padding=1)
        
        self.inception_4a = InceptionBlock(480, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['4a']))
        self.inception_4b = InceptionBlock(512, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['4b']))
        self.inception_4c = InceptionBlock(512, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['4c']))
        self.inception_4d = InceptionBlock(512, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['4d']))
        self.inception_4e = InceptionBlock(528, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['4e']))

        self.maxpool_4 = nn.MaxPool3d((2, 2, 2), stride=2)

        self.inception_5a = InceptionBlock(832, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['5a']))
        self.inception_5b = InceptionBlock(832, I3D.build_filter_specs(INCEPTION_BLOCK_FILTER_SPECS['5b']))
       
        self.avgpool_1 = nn.AvgPool3d((8, 7, 7))
        self.out_conv = nn.Conv3d(1024, out_channels, (1, 1, 1))

        #if pretrained_net is not None:
        #    self.load_pretrained_weights(pretrained_net)
    
    '''
    def load_pretrained_weights(self, net: models.googlenet.GoogLeNet):
        self.conv_1.conv.weight = nn.Parameter(net.conv1.conv.weight.repeat(7, 1, 1))
        self.conv_2.conv.weight = nn.Parameter(net.conv2.conv.weight.repeat(1, 1, 1))
        self.conv_2.conv.weight = nn.Parameter(net.conv2.conv.weight.repeat(1, 1, 1))
        pass
    '''
    
    def forward(self, x):
        c1_out = self.conv_1(x)
        m1_out = self.maxpool_1(c1_out)
        c2_out = self.conv_2(m1_out)
        c3_out = self.conv_3(c2_out)
        m2_out = self.maxpool_2(c3_out)

        i3a_out = self.inception_3a(m2_out)
        i3b_out = self.inception_3b(i3a_out)

        m3_out = self.maxpool_3(i3b_out)
        i4a_out = self.inception_4a(m3_out)
        i4b_out = self.inception_4b(i4a_out)
        i4c_out = self.inception_4c(i4b_out)
        i4d_out = self.inception_4d(i4c_out)
        i4e_out = self.inception_4e(i4d_out)

        m4_out = self.maxpool_4(i4e_out)

        i5a_out = self.inception_5a(m4_out) 
        i5b_out = self.inception_5b(i5a_out)
        a1_out = self.avgpool_1(i5b_out)
        f_out = self.out_conv(a1_out)
        out_logits = torch.squeeze(f_out)
        #out_logits = flatten(f_out)
        return out_logits

    @staticmethod
    def build_filter_specs(filter_list):
        spec_keys = ['1x1', '3x3_reduce1', '3x3_1', '3x3_reduce2', '3x3_2', 'pool_proj']
        return {k: v for k, v in zip(spec_keys, filter_list)}
    
