import torch
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn


def _make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                     kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))

class hand_model(torch.nn.Module):
    def __init__(self):
        super(hand_model, self).__init__()
        
        no_relu_layers = []
        vgg = OrderedDict([
                      ('conv1_1', [1, 64, 3, 1, 1]),
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
        self.vgg = _make_layers(vgg, no_relu_layers)
  
        self.L1 = Stage(128, 2)
        # self.L2 = Stage(148, 20) # 148 = 128+20
        # self.L3 = Stage(148, 20)
        # self.L4 = Stage(148, 20)
        # self.L5 = Stage(148, 20)
        # self.L6 = Stage(148, 20)

        self.S1 = Stage(130, 2, relu_last=True) 
        # self.S2 = Stage(159, 11, relu_last=True) # 159 = 128+20+11
        # self.S3 = Stage(159, 11, relu_last=True)
        # self.S4 = Stage(159, 11, relu_last=True)
        # self.S5 = Stage(159, 11, relu_last=True)
        # self.S6 = Stage(159, 11, relu_last=True)
        
    def forward(self, x):
        Fea = self.vgg(x)
        L1 = self.L1(Fea)
        S1 = self.S1(torch.cat([Fea, L1], dim=1))
        # L2 = self.L2(torch.cat([Fea, L1], dim=1))
        # S2 = self.S2(torch.cat([Fea, L2, S1], dim=1))
        # L3 = self.L3(torch.cat([Fea, L2], dim=1))
        # S3 = self.S3(torch.cat([Fea, L3, S2], dim=1))
        # L4 = self.L4(torch.cat([Fea, L3], dim=1))
        # S4 = self.S4(torch.cat([Fea, L4, S3], dim=1))
        # L5 = self.L5(torch.cat([Fea, L4], dim=1))
        # S5 = self.S5(torch.cat([Fea, L5, S4], dim=1))
        # L6 = self.L6(torch.cat([Fea, L5], dim=1))
        # S6 = self.S6(torch.cat([Fea, L6, S5], dim=1))    
        # return L1, L2, L3, L4, L5, L6, S1, S2, S3, S4, S5, S6
        return L1, S1

class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, relu_last = False):
        super(Stage, self).__init__()
        self.module1 = Inception(in_channels)
        self.module2 = Inception(384) # 128*3 = 384
        self.module3 = Inception(384)
        self.module4 = Inception(384)
        self.module5 = Inception(384)
        
        if not relu_last: 
            self.c1 = Conv_leaky( 384, 128, 1, 1, 0) # with leaky
            self.c2 = nn.Conv2d( 128, out_channels, 1, 1, 0, bias=True) # without relu
        else: 
            self.c1 = Conv(384, 128, 1, 1, 0)                       # with relu
            self.c2 = Conv( 128, out_channels, 1, 1, 0)             # with relu
    
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
class Conv_leaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_leaky, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.ReLU(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

if __name__ =='__main__':
    model = hand_model()
    img = torch.rand([2,1,360,360])
    print(img.shape)
    output = model(img)
    for i, dat in enumerate(output):
        print(i, dat.shape)