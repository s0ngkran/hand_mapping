import torch
import torch.nn.functional as F
def loss(out, gt_l, gt_s):
    width = gt_l.shape[2]
    height = gt_l.shape[3]
    loss = 0
    for L in range(6):
        pred = F.interpolate(out[L], (width, height) ,mode='bicubic')
        loss += ((pred - gt_l)**2).sum()
    for S in range(6):
        pred = F.interpolate(out[6+S], (width, height) ,mode='bicubic')
        loss += ((pred - gt_s)**2).sum()
    loss /= gt_l.shape[0] # divide by batch size
    return loss