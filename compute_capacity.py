from models.i3d import I3D
from models.c3d import C3D

import torch
import numpy as np

def conv_3d_cap(layer, input_per_bit):
    weight_bits = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * layer.in_channels
    input_bits = weight_bits * input_per_bit
    weight_bits += 1
    capacity = min(weight_bits, input_bits) * layer.out_channels
    return capacity

def conv_seq_cap(seq):
    cap = 0
    for l in seq:
        cap += conv_3d_cap(l.conv, 1)
    return cap

def inception_cap(block):
    branch_1x1_cap = conv_3d_cap(block.branch_1x1.conv, 1)
    branch_3x3_1_cap = conv_seq_cap(block.branch_3x3_1)
    branch_3x3_2_cap = conv_seq_cap(block.branch_3x3_2)
    branch_pool_cap = conv_3d_cap(block.branch_pool[1].conv, 1)
    return branch_1x1_cap + branch_3x3_1_cap + branch_3x3_2_cap + branch_pool_cap

def mock_input():
    return torch.randn(1, 3, 64, 224, 224)

def compute_I3D_cap(num_params_factor=1.0):
    inp = mock_input()

    i3d = I3D(3, 101, num_params_factor=num_params_factor)
    cap = {}
    
    min_input_size = 1e20   
    cap['conv1'] = min(min_input_size, conv_3d_cap(i3d.conv_1.conv, 8))
    

    inp = i3d.conv_1(inp)
    inp = i3d.maxpool_1(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['conv2'] = min(min_input_size, conv_3d_cap(i3d.conv_2.conv, 1))
    
    inp = i3d.conv_2(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['conv3'] = min(min_input_size, conv_3d_cap(i3d.conv_3.conv, 1))
    
    inp = i3d.conv_3(inp)
    inp = i3d.maxpool_2(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_3a'] = min(min_input_size, inception_cap(i3d.inception_3a))
    
    inp = i3d.inception_3a(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_3b'] = min(min_input_size, inception_cap(i3d.inception_3b))

    inp = i3d.inception_3b(inp)
    inp = i3d.maxpool_3(inp)

    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_4a'] = min(min_input_size, inception_cap(i3d.inception_4a))

    inp = i3d.inception_4a(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_4b'] = min(min_input_size, inception_cap(i3d.inception_4b))
    
    inp = i3d.inception_4b(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_4c'] = min(min_input_size, inception_cap(i3d.inception_4c))
    
    inp = i3d.inception_4c(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_4d'] = min(min_input_size, inception_cap(i3d.inception_4d))
    
    inp = i3d.inception_4d(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_4e'] = min(min_input_size, inception_cap(i3d.inception_4e))

    inp = i3d.inception_4e(inp)
    inp = i3d.maxpool_4(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_5a'] = min(min_input_size, inception_cap(i3d.inception_5a))
    
    inp = i3d.inception_5a(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['inception_5b'] = min(min_input_size, inception_cap(i3d.inception_5b))

    inp = i3d.inception_5b(inp)
    inp = i3d.avgpool_1(inp)
    min_input_size = min(np.prod(inp.size()), min_input_size)
    cap['out_conv'] = min(min_input_size, conv_3d_cap(i3d.out_conv, 1)) * 101
    return cap, sum(cap.values())

def compute_C3D_cap(num_params_factor=1.0):
    c3d = C3D(3, 101, num_params_factor=num_params_factor)
    cap = {}
    cap['conv1a'] = conv_3d_cap(c3d.conv_1a.conv, 8)
    cap['conv2a'] = conv_3d_cap(c3d.conv_2a.conv, 1)
    
    cap['conv3a'] = conv_3d_cap(c3d.conv_3a.conv, 1)
    cap['conv3b'] = conv_3d_cap(c3d.conv_3b.conv, 1)

    cap['conv4a'] = conv_3d_cap(c3d.conv_4a.conv, 1)
    cap['conv4b'] = conv_3d_cap(c3d.conv_4b.conv, 1)

    cap['conv5a'] = conv_3d_cap(c3d.conv_5a.conv, 1)
    cap['conv5b'] = conv_3d_cap(c3d.conv_5b.conv, 1)
    
    cap['fc6'] = conv_3d_cap(c3d.fc6.conv, 1)
    cap['fc7'] = conv_3d_cap(c3d.fc7, 1)
    
    return cap, sum(cap.values())

if __name__ == '__main__':
    print(compute_I3D_cap(1.0))
 #   print(compute_C3D_cap(1.0))
