from yolo_model import hand_model
import torch
# import torch.nn.functional as F
# import torch.nn as nn
import torch.optim as optim
import time as t
from loss_function import loss as lf
# import os

tr = [i for i in range(330)]
n_epochs = 10000
batch_size = 1
all_iter = len(tr)//batch_size

#learningRate = 0.01
learningRate = 0.000001/len(tr)
model = hand_model().cuda()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

# startepoch = 1
# loss = []
# times = []

checkpoint = torch.load('saveV4/train0018_ep10000490.pth') #waitinggggggggggggggg
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
startepoch = checkpoint['epoch']
loss = checkpoint['loss']
times = [checkpoint['time']]
print('fin import state')

epoch = startepoch
saveEvery = 20
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
        gt = []
        for i in indices:
            name = int(tr[i])
            namei = str(name).zfill(5)
           
            img_ = torch.load('../data011/imgi640_torch/'+namei).cuda()
            gt_ = torch.load('../data011/gt/'+namei).cuda()
            
            img.append(img_)
            gt.append(gt_)

        img = torch.stack(img)
        gt = torch.stack(gt)

        loss_ = lf(model(img),gt[0]) #check if batch
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
            }, 'saveV4/train0018_ep%d.pth'%(10000000+epoch))

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