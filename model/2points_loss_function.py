import torch
import torch.nn.functional as F
from Logger import Logger
logger = Logger('log_loss')
def loss_func(out, gt_l, gt_s):  #, covered):
    assert len(out) == 4, 'check your output stage'
    batch = gt_l.shape[0]
    width = gt_l.shape[2]
    height = gt_l.shape[3]
    loss = 0
    print_loss = []
    for i in range(3):
        pred_l = out[i]
        loss_ = ((pred_l - gt_l)**2)

        # for batch in range(gt_l.shape[0]):
        #     covered_point, covered_link = covered[batch]
        #     for i, cov in enumerate(covered_link):
        #         if cov:
        #             loss_[batch, i] = 0
        #     loss += loss_.sum()
        loss += loss_.sum()
    print_loss.append(loss)

    for i in range(1):
        pred_s = out[3]
        loss_ = ((pred_s - gt_s)**2)

        # for batch in range(gt_s.shape[0]):
        #     covered_point, covered_link = covered[batch]
        #     for i, cov in enumerate(covered_point):
        #         if cov:
        #             loss_[batch, i] = 0
        #     loss += loss_.sum()
        loss+= loss_.sum()
        print_loss.append(loss_.sum())

    msg = [str(i) for i in print_loss]
    logger.write('loss_L= %s loss_S= %s'%(msg[0],msg[1]))
    loss /= batch # divide by batch size
    return loss

if __name__ == '__main__':
    out = [torch.rand(2,2,60,60) for i in range(4)]
    gts = torch.rand(2,2,60,60)
    gtl = torch.rand(2,2,60,60)
    print('input len=', len(out), out[0].shape)
    loss = loss_func(out, gtl, gts)
    print('loss=', loss)
