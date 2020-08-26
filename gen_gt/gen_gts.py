import torch
from torch.nn import functional as F
import os 
import pickle
def gen_gts(cx,cy,width,height,sigma):
    # cx -> center x
    # cy -> center y
    # width, height of image
    # sigma -> size of heat
    emt = torch.zeros([width,height])
    x, y = torch.where(emt==0)
    distx = (x-cx).float()
    disty = (y-cy).float()
    dist = distx**2+disty**2
    ans = torch.exp(-(dist/sigma**2).float())
    ans = ans.reshape([width,height])
    
    # example code
    # import torch
    # import matplotlib.pyplot as plt
    # a = gen_gts(200,200,400,600,10)
    # plt.imshow(a)
    # plt.colorbar()
    # plt.show()
    return ans

def generate_gts(gt_folder, dist_folder, dim1, dim2, sigma):
    for _,__,gt_names in os.walk(gt_folder):
        print('fin walk')

    for i, gt_name in enumerate(gt_names):
        
        name = gt_name[:-4]
        with open(gt_folder + gt_name, 'rb') as f:
            data = pickle.load(f)
            data = data['keypoint']
        
        gts = []
        for x,y in data:
            cx,cy = x,y
            width,height = dim1
            gts_ = gen_gts(cx,cy,width,height,sigma)
            gts.append(gts_)
        gts = torch.stack(gts)

        gt = gts
        width, height = dim2
        gt = F.interpolate(gt.unsqueeze(0), (width, height) ,mode='bicubic')
        gt = gt.squeeze()

        torch.save(gt, dist_folder+name)
        print(name, i+1,len(gt_names))
        # plt.imshow(torch.max(a, dim=0)[0])
        # plt.show()
if __name__ == "__main__":
    gt_folder, dist_folder, dim1, dim2, sigma = 'testing/pkl/', 'testing/gts/', (360,360),(45,45),18
    generate_gts(gt_folder, dist_folder, dim1, dim2, sigma)
