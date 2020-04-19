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
