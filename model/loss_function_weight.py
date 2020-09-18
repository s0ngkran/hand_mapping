import torch
import torch.nn.functional as F
from Logger import Logger
logger = Logger('log_loss')
def loss_func(out, gt_l, gt_s):  #, covered):
    assert len(out) == 4, 'check your output stage'

    
    batch = gt_l.shape[0]
    width = gt_l.shape[2]
    height = gt_l.shape[3]
    shape = gt_l.shape
    loss = 0
    print_loss = []

    thres_zero = 1/99 *0.6
    thres_non  = 99/1 *0.4
    weight = torch.cuda.FloatTensor(shape).fill_(1)
    for i in range(3):
        pred_l = out[i]
        weight[gt_l==0] *= thres_zero
        weight[gt_l!=0] *= thres_non
        loss_ = weight.clone()*((pred_l - gt_l)**2)

        # for batch in range(gt_l.shape[0]):
        #     covered_point, covered_link = covered[batch]
        #     for i, cov in enumerate(covered_link):
        #         if cov:
        #             loss_[batch, i] = 0
        #     loss += loss_.sum()
        loss += torch.sum(loss_)

    thres = 0.01
    thres_zero = 4/96 *0.6
    thres_non  = 96/4 *0.4
    weight = torch.cuda.FloatTensor(shape).fill_(1)
    for i in range(1):
        pred_s = out[3]
        weight[gt_s < thres] *= thres_zero
        weight[gt_s >= thres] *= thres_non
        loss_ = weight.clone()*((pred_s - gt_s)**2)

        # for batch in range(gt_s.shape[0]):
        #     covered_point, covered_link = covered[batch]
        #     for i, cov in enumerate(covered_point):
        #         if cov:
        #             loss_[batch, i] = 0
        #     loss += loss_.sum()
        loss+= torch.sum(loss_)

    loss /= batch # divide by batch size
    return loss
def testloss():
    gts = torch.load('test_lossfunction/gts_test.torch').cuda()
    gtl = torch.load('test_lossfunction/gtl_test.torch').cuda()
    out = torch.stack([gtl,gtl+0.1,gtl,gts+0.1]).cuda()
    loss = loss_func(out, gtl, gts)
    print(loss)
def test_size(): # by rand
    out = [torch.rand(2,2,60,60) for i in range(4)]
    gts = torch.rand(2,2,60,60)
    gtl = torch.rand(2,2,60,60)
    print('input len=', len(out), out[0].shape)
    loss = loss_func(out, gtl, gts)
    print('loss=', loss)

if __name__ == '__main__':
    testloss()
