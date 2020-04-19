from hand_model import hand_model
import torch
# import torch.nn.functional as F
# import torch.nn as nn
import torch.optim as optim
import time as t
from loss_function import loss as lf
# import os

tr = torch.load('../data010/training_set.torch')
n_epochs = 10000
batch_size = 5
all_iter = len(tr)//batch_size

learningRate = 0.01
learningRate = 0.01/len(tr)
model = hand_model().cuda()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

# checkpoint = torch.load('save/train0012_ep10002796.pth') #waitinggggggggggggggg
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# startepoch = checkpoint['epoch']
# loss = checkpoint['loss']
# times = [checkpoint['time']]
# print('fin import state')

startepoch = 1
epoch = startepoch
loss = []
times = []

saveEvery = 5
iteration = 1
time_ = t.time()

for ep in range(n_epochs):
    permutation = torch.randperm(len(tr))
    for i in range(0,len(tr), batch_size):
        optimizer.zero_grad()
        ii = i+batch_size
        if ii>(len(tr)-1):break
        indices = permutation[i:ii]
        img = []
        gts = []
        gtl = []
        for i in indices:
            name = int(tr[i])
            namei = str(name).zfill(5)
            
            #isRev = True if tr[i][-4:] == '_rev' else False
           
            img_ = torch.load('../data010/Lhand640_torch/'+namei+'.jpg.torch').cuda()
            gts_ = torch.load('../data010/Lhand640_gts/'+namei).cuda()
            gtl_ = torch.load('../data010/Lhand640_gtl/'+namei).cuda()
            
            img.append(img_)
            gts.append(gts_)
            gtl.append(gtl_)

        img = torch.stack(img)
        gts = torch.stack(gts)
        gtl = torch.stack(gtl)

        loss_ = lf(model(img),gtl,gts)
        
        loss.append(loss_)
        loss_.backward()
        optimizer.step()
        
        #torch.save('this is a temp file.', 'log/epoch_%d__iter%d_%d__loss%d'%(epoch, iteration, all_iter, int(loss_)))
        print('epoch=',epoch,     'iter=',iteration,'/',all_iter, 'loss=',int(loss_))
        iteration += 1
    iteration = 1
    epoch += 1
    
        
    if epoch % saveEvery == 0 or epoch-startepoch <= 2 :
        thisEpochTime = t.time()-time_
        times.append(thisEpochTime)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'time': sum(times)
            }, 'save/train0017_ep%d.pth'%(10000000+epoch))

        print('\n\n\n\n\n')
        print('-------------------------------------------------------------')
        print('\n++++++saved++++++\n','epoch =', epoch, '   loss =',loss_)
        temp = saveEvery/thisEpochTime
        print('used time =', int(sum(times)/60), 'min')
        print('epochs in 10 min =', int(temp*60*10))
        print('epochs in 30 min =', int(temp*60*30))
        print('epochs in 1 hour =',int( temp*60*60))
        print('epochs in 2 hour =',int( temp*60*60*2))
        print('epochs in 4 hour =',int( temp*60*60*4))
        print('epochs in 1 day =',int( temp*60*60*24))
        print('epochs in 7 day =',int( temp*60*60*24*7))
        print('-------------------------------------------------------------')
        print('\n\n\n\n\n')
        time_ = t.time()
print('fin all')