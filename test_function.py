import os 
import torch 
from torch.nn import functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
class Table:
    def __init__(self):
        self.table = []
class Tester:
    def __init__(self, pkl_folder, n_part=2):
        assert pkl_folder[-1] == '/'
        for _,__,pkls in os.walk(pkl_folder):
            print('fin walk pkl_folder')
        pkls.sort()
        assert int(pkls[0][:10]) < int(pkls[1][:10]), 'check sorting method %s,%s'%(int(pkls[0][:10]), int(pkls[1][:10]))
        gt = []
        for pkl in pkls:
            with open(pkl_folder + pkl, 'rb') as f:
                data = pickle.load(f)['keypoint']
                gt.append([[data[i]] for i in range(len(data))]) 
        self.gt = gt # [img_index, part, [x,y]]
        assert self.gt != [], 'get empty groundtruth'
        self.n_part = n_part
        self.table = [Table() for i in range(n_part)]
        gt_ = np.array(gt)
        self.n_gt = [len(gt_[:, part]) for part in range(n_part)]
    def addtable(self, outs, indices, ratiofactor=0.3):
        peaks, confidences = self.getpeaks(outs) # batch, part, i, [x, y]
        gt = [self.gt[i-1] for i in indices] # batch, part, i, [x, y]
        ratio = [self.getdistance(gt[batch][0][0], gt[batch][1][0])*ratiofactor for batch in range(len(gt))] # batch
    
        for batch in range(len(peaks)):
            peaks_ = peaks[batch]
            gt_ = gt[batch]
            ratio_ = ratio[batch]
            img_index = indices[batch]
            confidence = confidences[batch]

            for part in range(self.n_part):
                # sort confidence to check high confidence first
                confidence_sorted = np.array(confidence[part]).argsort()[::-1]
                for ind in confidence_sorted: # old code
                    pred = peaks_[part][ind]
                    for ind_gt in range(len(gt_[part])):
                        gt_ind = -1
                        center = gt_[part][ind_gt]
                        correct = self.inPCK(pred, center, ratio_)
                        if correct:
                            gt_ind = ind_gt
                            break
                    self.table[part].table.append([img_index, float(confidence[part][ind]), gt_ind, correct])
    def getacc(self, filename):
        self.ap = []
        self.auc = []
        for part in range(self.n_part):
            self.table[part].sort(key=lambda x:x[1], reverse= True)
            table = self.table[part]
            n_gt = self.n_gt[part]
            for i in range(len(table)): #check same gt
                img_ind = table[i][0]
                for j in range(len(table)-i-1):
                    j = i + j+1
                    img_ind2 = table[j][0]
                    if img_ind == img_ind2:
                        gt_ind = table[i][2]
                        gt_ind2 = table[j][2]
                        if gt_ind == gt_ind2:
                            if table[i][3] == True:
                                table[j][3] = False
            acc_tp = 0
            acc_fp = 0
            ap_point = []
            for i in range(len(table)):
                if table[i][3]:
                    acc_tp += 1
                pr = acc_tp/(i+1)
                re = acc_tp/n_gt
                ap_point.append([pr,re])
            self.ap.append(ap_point)
            auc = 0
            for i in range(len(ap_point)):
                if i+1 < len(ap_point):
                    recall_ = ap_point[i+1][1]-ap_point[i][1]
                    precision_ = (ap_point[i][0] + ap_point[i+1][0])/2
                    auc += recall_ * precision_
            self.auc.append(auc)
        # save section
        self.save(filename)
        return self.auc
    def save(self, filename=''):
        if filename == '': filename = 'savetemp_.torch'
        table = [self.table[i] for i in range(self.n_part)]
        ap = self.ap
        auc = self.auc
        torch.save({'table':table
                    ,'AP':ap
                    ,'auc':auc }, filename)
    def inPCK(self, pred, gt, ratio):
        dist = self.getdistance(pred, gt)
        return True if dist < ratio else False
    def getdistance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2 
        dist = (x1-x2)**2 + (y1-y2)**2 
        return dist**0.5
    def getpeaks(self, outs, size=(360,360), thres=0.1):
        pred_L = outs[0]
        pred_S = outs[1]
        batch = pred_L.shape[0]
        n_L = pred_L.shape[1]
        n_S = pred_S.shape[1]
        assert len(pred_L.shape) == 4, 'recheck out shape => [batch, parts, x, y]'
        assert pred_L.shape[1] in [2, 20], 'link should be 2 or 20'

        # pred_L = F.interpolate(pred_L, size, mode='bicubic')
        pred_S = F.interpolate(pred_S, size, mode='bicubic')
        shape = pred_S.shape
        map_left  = torch.zeros(shape)
        map_right = torch.zeros(shape)
        map_up    = torch.zeros(shape)
        map_down  = torch.zeros(shape)
        map_left2  = torch.zeros(shape)
        map_right2 = torch.zeros(shape)
        map_up2    = torch.zeros(shape)
        map_down2  = torch.zeros(shape)
        n_ck = 1
        map_left[:,:,n_ck:, :]   = pred_S[:, :,:-n_ck, :  ]
        map_right[:,:,:-n_ck, :] = pred_S[:,:,n_ck:  , :  ]
        map_up[:,:,:, n_ck:]     = pred_S[:,:, :  , :-n_ck]
        map_down[:,:,:, :-n_ck]  = pred_S[:,:, :  ,n_ck:  ]
        n_ck = 2
        map_left2[:,:,n_ck:, :]   = pred_S[:, :,:-n_ck, :  ]
        map_right2[:,:,:-n_ck, :] = pred_S[:,:,n_ck:  , :  ]
        map_up2[:,:,:, n_ck:]     = pred_S[:,:, :  , :-n_ck]
        map_down2[:,:,:, :-n_ck]  = pred_S[:,:, :  ,n_ck:  ]
        pred_S_max = torch.BoolTensor ((pred_S >= map_left )&
                                    (pred_S >= map_right )&
                                    (pred_S >= map_up)&
                                    (pred_S >= map_down)&
                                    (pred_S >= map_left2 )&
                                    (pred_S >= map_right2 )&
                                    (pred_S >= map_up2)&
                                    (pred_S >= map_down2)&
                                    (pred_S >= thres)  )
        
        pred = [[torch.nonzero(pred_S_max[batch_, part_]) for part_ in range(n_S)] for batch_ in range(batch)]
        # pred = [batch][part][i][x, y]

        # test
        # print('pred', pred[1][1])
        # plt.imshow(pred_S[1][1].transpose(0,1))
        # for x,y in pred[1][1]:
        #     plt.plot(x,y,'r.')
        # plt.show()
        confidence = [] # confidence = [batch][part][i][value]
        for batch_ in range(batch):
            con_batch = []
            for part_ in range(n_S):
                con_part = []
                for i in range(len(pred[batch_][part_])):
                    x, y = pred[batch_][part_][i]
                    con_ = pred_S[batch_, part_, x, y]
                    con_part.append(con_)
                con_batch.append(con_part)
            confidence.append(con_batch)
   
        return pred, confidence

def test_getacc():
    # from hand_model1 import hand_model 
    import torch 
    # x = torch.rand([2,1,360,360])
    # model = hand_model()
    # out = model(x)
    gts1 = torch.load('temp/0000000002_2ps')
    gts2 = torch.load('temp/0000000004_2ps')
    gts = torch.stack([gts1, gts2])
    print('gts =',gts.shape)

    gtl1 = torch.load('temp/0000000002_2p')
    gtl2 = torch.load('temp/0000000004_2p')
    gtl = torch.stack([gtl1, gtl2])
    print('gtl =',gtl.shape)

    out = (gtl, gts)

    # tester = Tester('../data014/testing/pkl/')
    tester = Tester('../data014/random_background/temp/')
    acc = tester.addtable(out, [2,4])
    print(tester.table[0].table)
    print(tester.table[1].table)

if __name__ == "__main__":
    test_getacc()
