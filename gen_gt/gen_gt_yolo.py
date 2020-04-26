import torch
import numpy as np 
import os 

cut = [i/7 for i in range(7)]
def grid(a):
    for ii in range(len(cut)):
        if a < cut[ii]:
            ans = ii-1
            break
    if a > cut[6]:
        ans = 6
    return ans

for _,__,peak_names in os.walk('peak/'):
    print('fin') 

for i in range(len(peak_names)):
    namei = str(i).zfill(5)
    peak = np.load('peak/'+peak_names[i])
    gt = torch.zeros([7,7,14])
    if peak[0] == 'blank':
        print('blank')
    else:
        #this program just for only 2 objs
        layer1 = 0
        for obj in peak:
            x = obj[0]
            y = obj[1]
            w = obj[2]
            h = obj[3]
            c = obj[4]
            class1 = obj[5]
            class2 = obj[6]

            gx = grid(x)
            gy = grid(y)

            if layer1 == 0:
                gt[gx,gy,:7] = torch.FloatTensor(obj)
                layer1 = 1
            else :
                gt[gx,gy,7:] = torch.FloatTensor(obj)

    torch.save(gt,'gt/'+namei)
    print(namei)
