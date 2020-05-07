import torch
import torch.nn as nn
import torch.nn.functional as func

class hand_model(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d (in_channels, out_channels, kernel_size, stride, padding)
        self.layer1 = nn.Conv2d(3,64,7,2,3)
        self.maxpool1 = nn.MaxPool2d(2,2,0)

        self.layer2 = nn.Conv2d(64,192,3,1,1)
        self.maxpool2 = nn.MaxPool2d(2,2,0)

        self.layer31 = nn.Conv2d(192,128,1,1,0)
        self.layer32 = nn.Conv2d(128,256,3,1,1)
        self.layer33 = nn.Conv2d(256,256,1,1,0)
        self.layer34 = nn.Conv2d(256,512,3,1,1)
        self.maxpool3 = nn.MaxPool2d(2,2,0)

        self.layer41 = nn.Conv2d(512,256,1,1,0)
        self.layer42 = nn.Conv2d(256,512,3,1,1)
        self.layer43 = nn.Conv2d(512,512,1,1,0)
        self.layer44 = nn.Conv2d(512,1024,3,1,1)
        self.maxpool4 = nn.MaxPool2d(2,2,0)

        self.layer51 = nn.Conv2d(1024,512,1,1,0)
        self.layer52 = nn.Conv2d(512,1024,3,1,1)
        self.layer53 = nn.Conv2d(1024,1024,3,1,1)
        self.layer54 = nn.Conv2d(1024,1024,3,2,1)

        self.layer61 = nn.Conv2d(1024,1024,3,1,1)
        self.dropout1 = nn.Dropout2d(0.5)

        self.layer7 = nn.Linear(61440,4096)
        self.layer8 = nn.Linear(4096,7*7*14)

        torch.nn.init.kaiming_normal_(self.layer1.weight)
        torch.nn.init.kaiming_normal_(self.layer2.weight)
        torch.nn.init.kaiming_normal_(self.layer31.weight)
        torch.nn.init.kaiming_normal_(self.layer32.weight)
        torch.nn.init.kaiming_normal_(self.layer33.weight)
        torch.nn.init.kaiming_normal_(self.layer34.weight)
        torch.nn.init.kaiming_normal_(self.layer41.weight)
        torch.nn.init.kaiming_normal_(self.layer42.weight)
        torch.nn.init.kaiming_normal_(self.layer43.weight)
        torch.nn.init.kaiming_normal_(self.layer44.weight)
        torch.nn.init.kaiming_normal_(self.layer51.weight)
        torch.nn.init.kaiming_normal_(self.layer52.weight)
        torch.nn.init.kaiming_normal_(self.layer53.weight)
        torch.nn.init.kaiming_normal_(self.layer54.weight)
        torch.nn.init.kaiming_normal_(self.layer61.weight)

    def forward(self, inp):

        a = self.layer1(inp)
        a = func.leaky_relu(a, 0.1)
        a = self.maxpool1(a)

        a = self.layer2(a)
        a = func.leaky_relu(a, 0.1)
        a = self.maxpool2(a)
        a = self.layer31(a)
        a = func.leaky_relu(a, 0.1)
        a = self.layer32(a)
        a = func.leaky_relu(a, 0.1)
        a = self.layer33(a)
        a = func.leaky_relu(a, 0.1)

        a = self.layer34(a)
        a = func.leaky_relu(a, 0.1)
        a = self.maxpool3(a)
        for i in range(4):
            a = self.layer41(a)
            a = func.leaky_relu(a, 0.1)
            a = self.layer42(a)
            a = func.leaky_relu(a, 0.1)
        a = self.layer43(a)
        a = func.leaky_relu(a, 0.1)
        a = self.layer44(a)
        a = func.leaky_relu(a, 0.1)
        a = self.maxpool4(a)

        for i in range(2):
            a = self.layer51(a)
            a = func.leaky_relu(a, 0.1)
            a = self.layer52(a)
            a = func.leaky_relu(a, 0.1)
        a = self.layer53(a)
        a = func.leaky_relu(a, 0.1)
        a = self.layer54(a)
        a = func.leaky_relu(a, 0.1)

        for i in range(2):
            a = self.layer61(a)
            a = func.leaky_relu(a, 0.1)

        a = self.dropout1(a)
        a = torch.flatten(a,1)
        #print(a.shape)
        a = self.layer7(a)
        a = self.layer8(a)
        a = a.reshape(7,7,14)
        a = func.relu(a)
        return a
        
       

