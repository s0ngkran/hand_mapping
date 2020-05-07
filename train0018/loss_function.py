import torch
def loss(output, gt):
    # output = torch.FloatTensor(output)
    # gt = torch.FloatTensor(gt)
    coord = 5
    noobj = 0.5
    loss_ = 0
    ss1 = gt[:,:,0]
    
    ss2 = gt[:,:,6]
    d1,d2 = torch.where(ss1 != 0)
    e1,e2 = torch.where(ss2 != 0)
    f1,f2 = torch.where(ss1 == 0)
    g1,g2 = torch.where(ss2 == 0)
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    temp5 = 0
    if len(d1)>0:
        for i in range(len(d1)):
            temp1 += (output[d1[i],d2[i],0]-gt[d1[i],d2[i],0])**2
            temp1 += (output[d1[i],d2[i],1]-gt[d1[i],d2[i],1])**2
            # temp2 += ((output[d1[i],d2[i],2])-(gt[d1[i],d2[i],2]))**2
            # temp2 += ((output[d1[i],d2[i],3])-(gt[d1[i],d2[i],3]))**2
            temp2 += (torch.sqrt(output[d1[i],d2[i],2])-torch.sqrt(gt[d1[i],d2[i],2]))**2
            temp2 += (torch.sqrt(output[d1[i],d2[i],3])-torch.sqrt(gt[d1[i],d2[i],3]))**2
            temp3 += (output[d1[i],d2[i],4]-gt[d1[i],d2[i],4])**2

            sum_p1 = torch.exp(output[d1[i],d2[i],5])+torch.exp(output[d1[i],d2[i],6])
            sum_p1gt = torch.exp(gt[d1[i],d2[i],5])+torch.exp(gt[d1[i],d2[i],6])
            temp5 += (output[d1[i],d2[i],5]/sum_p1 - gt[d1[i],d2[i],5]/sum_p1gt)**2
            temp5 += (output[d1[i],d2[i],6]/sum_p1 - gt[d1[i],d2[i],6]/sum_p1gt)**2
    if len(e1)>0:
        for i in range(len(e1)):
            temp1 += (output[e1[i],e2[i],0]-gt[e1[i],e2[i],0])**2
            temp1 += (output[e1[i],e2[i],1]-gt[e1[i],e2[i],1])**2
            # temp2 += ((output[d1[i],d2[i],2])-(gt[d1[i],d2[i],2]))**2
            # temp2 += ((output[d1[i],d2[i],3])-(gt[d1[i],d2[i],3]))**2
            temp2 += (torch.sqrt(output[e1[i],e2[i],2])-torch.sqrt(gt[e1[i],e2[i],2]))**2
            temp2 += (torch.sqrt(output[e1[i],e2[i],3])-torch.sqrt(gt[e1[i],e2[i],3]))**2
            temp3 += (output[e1[i],e2[i],4]-gt[e1[i],e2[i],4])**2

            sum_p2 = torch.exp(output[e1[i],e2[i],12])+torch.exp(output[e1[i],e2[i],13])
            sum_p2gt = torch.exp(gt[e1[i],e2[i],12])+torch.exp(gt[e1[i],e2[i],13])
            temp5 += (output[e1[i],e2[i],12]/sum_p1 - gt[e1[i],e2[i],12]/sum_p1gt)**2
            temp5 += (output[e1[i],e2[i],13]/sum_p1 - gt[e1[i],e2[i],13]/sum_p1gt)**2
    for i in range(len(f1)): #noobj confidence
        temp4 += (output[f1[i],f2[i],4]-gt[f1[i],f2[i],4])**2
    for i in range(len(g1)): #noobj confidence
        temp4 += (output[g1[i],g2[i],4]-gt[g1[i],g2[i],4])**2

    # print('temp1=',float(temp1))
    # print('temp2=',float(temp2))
    # print('temp3=',float(temp3))
    # print('temp4=',float(temp4))
    # print('temp5=',float(temp5))
    loss_ = coord*temp1 + coord*temp2 + temp3 + noobj*temp4 + temp5
    return loss_
   

    #loss /= gt.shape[0] # divide by batch size
   