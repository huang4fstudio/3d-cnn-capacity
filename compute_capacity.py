from models.i3d import I3D
from models.c3d import C3D

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

def compute_I3D_cap(num_params_factor=1.0):
    i3d = I3D(3, 101, num_params_factor=num_params_factor)
    cap = {}
    cap['conv1'] = conv_3d_cap(i3d.conv_1.conv, 8)
    cap['conv2'] = conv_3d_cap(i3d.conv_2.conv, 1)
    cap['conv3'] = conv_3d_cap(i3d.conv_3.conv, 1)

    cap['inception_3a'] = inception_cap(i3d.inception_3a)
    cap['inception_3b'] = inception_cap(i3d.inception_3b)

    cap['inception_4a'] = inception_cap(i3d.inception_4a)
    cap['inception_4b'] = inception_cap(i3d.inception_4b)
    cap['inception_4c'] = inception_cap(i3d.inception_4c)
    cap['inception_4d'] = inception_cap(i3d.inception_4d)
    cap['inception_4e'] = inception_cap(i3d.inception_4e)

    cap['inception_5a'] = inception_cap(i3d.inception_5a)
    cap['inception_5a'] = inception_cap(i3d.inception_5b)

    cap['out_conv'] = conv_3d_cap(i3d.out_conv, 1)
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
    print(compute_I3D_cap(0.5))
    print(compute_C3D_cap(0.5))