import torch
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn


def _make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = torch.nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = torch.nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                     kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, torch.nn.ReLU(inplace=True)))
    return torch.nn.Sequential(OrderedDict(layers))

class hand_model(torch.nn.Module):
    
    def __init__(self):
        super(hand_model, self).__init__()
        
        no_relu_layers = []
        vgg = OrderedDict({
            'conv1_1': [3, 64, 3, 1, 1],
            'conv1_2': [64, 64, 3, 1, 1],
            'pool1_stage1': [2, 2, 0],
            'conv2_1': [64, 128, 3, 1, 1],
            'conv2_2': [128, 128, 3, 1, 1],
            'pool2_stage1': [2, 2, 0],
            'conv3_1': [128, 256, 3, 1, 1],
            'conv3_2': [256, 256, 3, 1, 1],
            'conv3_3': [256, 256, 3, 1, 1],
            'conv3_4': [256, 256, 3, 1, 1],
            'pool3_stage1': [2, 2, 0],
            'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1],
            'conv4_3_CPM': [512, 256, 3, 1, 1],
            'conv4_4_CPM': [256, 128, 3, 1, 1]
        })
        self.vgg = _make_layers(vgg, no_relu_layers)      
        self.L1 = Stage(128, 10)
        self.L2 = Stage(139, 10) # 139 = 128+11
        self.L3 = Stage(139, 10)
        self.L4 = Stage(139, 10)
        self.L5 = Stage(139, 10)
        self.L6 = Stage(139, 10)

        self.S1 = Stage(138, 11) # 138 = 128+10
        self.S2 = Stage(138, 11)
        self.S3 = Stage(138, 11)
        self.S4 = Stage(138, 11)
        self.S5 = Stage(138, 11)
        self.S6 = Stage(138, 11)
    def do(self, stage, x):
        x = stage(x)
        return x, torch.cat([self.fea, x],dim=1)
    def forward(self, x):
        self.fea = self.vgg(x)
        L1 = self.L1(self.fea)
        x = torch.cat([self.fea, L1],dim=1)

        S1, x = self.do(self.S1, x)
        L2, x = self.do(self.L2, x)
        S2, x = self.do(self.S2, x)
        L3, x = self.do(self.L3, x)
        S3, x = self.do(self.S3, x)
        L4, x = self.do(self.L4, x)
        S4, x = self.do(self.S4, x)
        L5, x = self.do(self.L5, x)
        S5, x = self.do(self.S5, x)
        L6, x = self.do(self.L6, x)
        S6, x = self.do(self.S6, x)
    
        return L1, L2, L3, L4, L5, L6, S1, S2, S3, S4, S5, S6
class Stage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stage, self).__init__()
        self.module1 = Inception(in_channels)
        self.module2 = Inception(384) # 128*3 = 384
        self.module3 = Inception(384)
        self.module4 = Inception(384)
        self.module5 = Inception(384)
        self.c1 = Conv(384, 128, 1, 1, 0)
        self.c2 = Conv(128, out_channels, 1, 1, 0)
    
    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5(x)
        x = self.c1(x)
        x = self.c2(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.c1 = Conv(in_channels, 128, 3, 1, 1)
        self.c2 = Conv(128, 128, 3, 1, 1)
        self.c3 = Conv(128, 128, 3, 1, 1)
    def forward(self, x):
        y1 = self.c1(x)
        y2 = self.c2(y1)
        y3 = self.c3(y2)
        return torch.cat([y1, y2, y3], dim=1)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

if __name__ =='__main__':
    model = hand_model()
    img = torch.rand([2,3,480,480])
    output = model(img)
    for i, dat in enumerate(output):
        print(i, dat.shape)