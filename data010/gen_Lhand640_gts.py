import numpy as np 
import torch
import matplotlib.pyplot as plt 
import os

def gen_gts(cx,cy,width,height,sigma):
    emt = torch.zeros([width,height])
    x, y = torch.where(emt==0)
    distx = (x-cx).float()
    disty = (y-cy).float()
    dist = distx**2+disty**2
    ans = torch.exp(-(dist/sigma**2))
    ans = ans.reshape([width,height])
    return ans

path = 'peakLhand640/'
for _,__,peak_names in os.walk(path):
    print('fin peak_names')

for i in range(len(peak_names)):
    namei = str(i+1).zfill(5)
    width = 640
    height = 360
    peak = np.load(path+peak_names[i])

    if peak[0]=='99999':
        gts = torch.zeros([11,width, height])
    else:
        for hand in peak:
            gts_ = []
            for point in hand:
                # c -> center
                cx = point[0]
                cy = point[1]
                tem = gen_gts(cx,cy,width,height,8)
                gts_.append(tem)
            gts = torch.stack(gts_)
    savepath = 'Lhand640_gts/'
    torch.save(gts,savepath+namei)
    print(namei)
print('fin all')
# a = gts.max(0)[0]
# plt.imshow(a.transpose(1,0))
# plt.show()


