import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as func

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            torch.nn.init.kaiming_normal_(conv2d.weight)
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class hand_model(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = []
        block_vgg = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])    ])

        self.vgg = make_layers(block_vgg, no_relu_layers)

        # Conv2d (in_channels, out_channels, kernel_size, stride, padding)
        
        self.conv_128 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv_148 = nn.Conv2d(148, 128, 3, 1, 1) # 128+20
        self.conv_159 = nn.Conv2d(159, 128, 3, 1, 1) # 128+20+11
        self.conv_384 = nn.Conv2d(384, 128, 3, 1, 1) # 128+128+128
        
        self.c_ = nn.Conv2d(384, 128, 1, 1, 0)
        self.c_20 = nn.Conv2d(128, 20, 1, 1, 0) # for L >>> PAF
        self.c_11 = nn.Conv2d(128, 11, 1, 1, 0) # for S >>> Heat

        torch.nn.init.kaiming_normal_(self.conv_128.weight)
        torch.nn.init.kaiming_normal_(self.conv_148.weight)
        torch.nn.init.kaiming_normal_(self.conv_159.weight)
        torch.nn.init.kaiming_normal_(self.conv_384.weight)
        torch.nn.init.kaiming_normal_(self.c_.weight)
        torch.nn.init.kaiming_normal_(self.c_20.weight)
        torch.nn.init.kaiming_normal_(self.c_11.weight)

    def block_128(self, input):
        t1 = func.relu(self.conv_128(input))
        t2 = func.relu(self.conv_128(t1))
        t3 = func.relu(self.conv_128(t2))
        return torch.cat([t1, t2, t3], 1)

    def block_148(self, input): # 128+20
        t1 = func.relu(self.conv_148(input))
        t2 = func.relu(self.conv_128(t1))
        t3 = func.relu(self.conv_128(t2))
        return torch.cat([t1, t2, t3], 1)
    
    def block_159(self, input): # 128+20+11
        t1 = func.relu(self.conv_159(input))
        t2 = func.relu(self.conv_128(t1))
        t3 = func.relu(self.conv_128(t2))
        return torch.cat([t1, t2, t3], 1)
    
    def block_384(self, input): # 128+128+128
        t1 = func.relu(self.conv_384(input))
        t2 = func.relu(self.conv_128(t1))
        t3 = func.relu(self.conv_128(t2))
        return torch.cat([t1, t2, t3], 1)

    def block_(self, input): # block_384 (4 times)
        for _ in range(4):
            input = self.block_384(input)
        return input

    def forward(self, x):

        F = self.vgg(x)

        t = self.block_128(F)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        L1 = self.c_20(t)

        C = torch.cat([L1, F], 1)
        t = self.block_148(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        S1 = self.c_11(t)

        C = torch.cat([L1, F], 1)
        t = self.block_148(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        L2 = self.c_20(t)

        C = torch.cat([L2, S1, F], 1)
        t = self.block_159(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        S2 = self.c_11(t)

        C = torch.cat([L2, F], 1)
        t = self.block_148(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        L3 = self.c_20(t)

        C = torch.cat([L3, S2, F], 1)
        t = self.block_159(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        S3 = self.c_11(t)

        C = torch.cat([L3, F], 1)
        t = self.block_148(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        L4 = self.c_20(t)

        C = torch.cat([L4, S3, F], 1)
        t = self.block_159(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        S4 = self.c_11(t)

        C = torch.cat([L4, F], 1)
        t = self.block_148(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        L5 = self.c_20(t)

        C = torch.cat([L5, S4, F], 1)
        t = self.block_159(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        S5 = self.c_11(t)

        C = torch.cat([L5, F], 1)
        t = self.block_148(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        L6 = self.c_20(t)

        C = torch.cat([L6, S5, F], 1)
        t = self.block_159(C)
        t = self.block_(t)
        t = func.relu(self.c_(t))
        S6 = self.c_11(t)
        return L1, L2, L3, L4, L5, L6, S1, S2, S3, S4, S5, S6