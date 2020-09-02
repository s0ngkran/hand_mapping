from hand_model1 import hand_model
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.optim as optim
import time as t
import os
from Logger import Logger
from line import Line
from test_function import Tester
import torch2img
from loss_function import loss_func as lf
import pickle
import torch.nn.functional as F
import requests
def googlesheet(epoch, tr, va, te1, te2):
    do = False
    if do:
        success = False
        while not success:
            # try:
                sheet = 'Sheet1'
                epoch_ = str(epoch)
                tr_ = '%.5f'%tr
                va_ = '%.5f'%va
                te_1 = '%.5f'%te1 if type(te1) != str else '%s'%te1
                te_2 = '%.5f'%te2 if type(te2) != str else '%s'%te2
                print('fin convert',sheet,epoch_,tr_,va_,te_1,te_2)
                url = 'https://script.google.com/macros/s/AKfycbxaHwYf8LOE3SIRznIGWWb2_nh5XN__DFFQCGbfZBWUIX1tREE/exec'
                res = requests.post(url, data={'sheet':sheet
                                        ,'epoch':epoch_
                                        ,'training':tr_
                                        ,'validation':va_
                                        ,'testing1':te_1 
                                        ,'testing2':te_2     })
                success = True
            # except:
            #     import time
            #     print('connection to googlesheet error, wait 10 sec')
            #     time.sleep(10)
        return res
def run(sed):
    line = Line(os.path.basename(__file__), turnoff=False)
    train_n = os.path.basename(__file__).split('train')[1].split('.py')[0]
    for seed in range(1):
        seed += sed
        torch.manual_seed(seed)
        model = hand_model().cuda()
        optimizer = optim.Adam(model.parameters())
        
        startepoch = 1
        times = []

        ######################### start config ###################################
        
        googlesheet('starting',0,0,0,0)
        logfile = 'tr%s'%(train_n)
        # savefolder = 'save%s_1/'%train_n
        savefolder = 'save/'
        logger = Logger(logfile)            #
        logger.write('start...')            #

        training_set_folder = '../data014/training/img_torch/random_background/'
        gts_folder = '../data014/training/gts/random_background/'
        gtl_folder = '../data014/training/gtl/random_background/'
        covered_tr = '../data013/covered/training.pkl' # no need

        va_folder = '../data014/validation/img_torch/'
        gts_va_folder = '../data014/validation/gts/'
        gtl_va_folder = '../data014/validation/gtl/'
        covered_va = '../data013/covered/validation.pkl' # no need

        te_folder = '../data014/testing/img_torch/'
        gts_te_folder = '../data014/testing/gts/'
        gtl_te_folder = '../data014/testing/gtl/'
        covered_te = '../data013/covered/testing.pkl' # no need

        ss1 = ['training/random_background/','validation/','testing/']
        ss2 = ['training/gts/random_background/','validation/gts/','testing/gts/']
        ss3 = ['training/gtl/random_background/','validation/gtl/','testing/gtl/']
        ex_folder = ['../data014/ex_img_torch/'+i for i in ss1]
        ex_gts_fol = ['../data014/'+i for i in ss2]
        ex_gtl_fol = ['../data014/'+i for i in ss3]

        # checkpoint = torch.load('save/train01_epoch0000000490_seed1.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])           #
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   #
        # startepoch = checkpoint['epoch']+1                              #
        # times = [checkpoint['time']]                                    #
        # print('fin import state')                                       #

        for _,__,training_set in os.walk(training_set_folder):          #
            logger.write('fin tr_set walk')                             #
        # training_set = training_set[:datasize]
        # param
        batch_size = 5
        all_iter = len(training_set)//batch_size
        learningRate = 0.0001 / all_iter
        index_predL, index_predS = 2, 3
        saveEvery = 10
        va_every = 5
        te_every = va_every
        line_every = 30 #epoch
        

        # mode
        train = 1
        train2ep = 3000
        validate = 1
        ex_img = 1
        test = 1
        ################################## end config ##############################################

        for param_group in optimizer.param_groups:
            param_group['lr'] = learningRate

        for _,__,validation_set in os.walk(va_folder):
            logger.write('fin va_set walk')
        with open(covered_va, 'rb') as f:
            covered_va = pickle.load(f)
        for _,__,testing_set in os.walk(te_folder):
            logger.write('fin te_set walk')
        with open(covered_te, 'rb') as f:
            covered_te = pickle.load(f)
        with open(covered_tr, 'rb') as f:
            covered_tr = pickle.load(f)

        epoch = startepoch
        time_ = t.time()
        logger.write('starting epoch%d'%epoch)
        
        while epoch<train2ep:
            if train:
                if epoch <2: print('start train')
                permutation = torch.randperm(len(training_set))
                iteration = 0
                for i in range(0, len(training_set), batch_size):
                    optimizer.zero_grad()
                    ii = i+batch_size
                    if ii>(len(training_set)-1):break
                    indices = permutation[i:ii]
                    img = []
                    gts = []
                    gtl = []
                    covered = []
                    for ind in indices: # i = index of tr
                        img_name = training_set[ind]
                        namei = img_name[:10] + '_2p'
                        # nameint = int(namei)
                        img_ = torch.load(training_set_folder + img_name)
                        gts_ = torch.load(gts_folder + namei)
                        gtl_ = torch.load(gtl_folder + namei)
                        img.append(img_)
                        gts.append(gts_)
                        gtl.append(gtl_)
                        # covered.append(covered_tr[nameint])# no need
                    img = torch.stack(img).cuda()
                    gts = torch.stack(gts).cuda()
                    gtl = torch.stack(gtl).cuda()

                    model.train()
                    out = model(img)
                    
                    loss = lf(out, gtl, gts)# , covered)
                    # loss.append(loss_)
                    loss.backward()
                    optimizer.step()
                    iteration += 1

                #     if iteration%100 == 0:
                #         logger.write('epoch=%d iter=%d loss=%.2f'%(epoch, iteration, loss)) # each iteration
                #     if iteration%1000 == 0:
                #         line.send('epoch=%d iter=%d loss=%.2f'%(epoch, iteration, loss))
                # if iteration%100 != 0:
                #     logger.write('epoch=%d loss=%.2f'%(epoch, loss)) # each iteration
                logger.write('epoch=%d loss=%.5f'%(epoch, loss))

                if epoch % line_every == 0 or epoch == 1:
                    line.send('epoch=%d loss=%.5f'%(epoch, loss))
            
            if validate and (epoch % va_every == 0 or epoch == 1):
                if epoch <2: print('start validate')
                permutation = torch.randperm(len(validation_set))
                va_loss = []
                for i in range(0,len(validation_set), batch_size):
                    ii = i+batch_size
                    if ii>len(validation_set): ii = len(validation_set)
                    indices = permutation[i:ii]
                    img = []
                    gts = []
                    gtl = []
                    covered = []
                    for ind in indices: 
                        img_name = validation_set[ind]
                        namei = img_name[:10] + '_2p'
                        # nameint = int(namei) # no need

                        img_ = torch.load(va_folder + img_name)
                        gts_ = torch.load(gts_va_folder + namei)
                        gtl_ = torch.load(gtl_va_folder + namei)
                        img.append(img_)
                        gts.append(gts_)
                        gtl.append(gtl_)
                        # covered.append(covered_va[nameint])
                    img = torch.stack(img).cuda()
                    gts = torch.stack(gts).cuda()
                    gtl = torch.stack(gtl).cuda()
                    with torch.no_grad():
                        model.eval()
                        out = model(img)
                        va_loss_ = lf(out, gtl, gts) #, covered) no need
                    va_loss.append(va_loss_)
                va_loss = sum(va_loss)/len(va_loss)

                logger.write('epoch=%d loss_va=%.5f'%(epoch, va_loss) )

                if epoch % line_every == 0 or epoch == 1:
                    line.send('epoch=%d loss_va=%.5f'%(epoch, va_loss) )
 
            if test and (epoch % te_every == 0 or epoch == 1):
                if epoch <2: print('start test')
                tester = Tester('../data014/testing/pkl/')
                permutation = torch.randperm(len(testing_set))
                te_loss = []
                for i in range(0,len(testing_set), batch_size):
                    ii = i+batch_size
                    if ii>len(testing_set): ii = len(testing_set)
                    indices = permutation[i:ii]
                    img = []
                    for ind in indices: 
                        img_name = testing_set[ind]
                        namei = img_name[:10] + '_2p'
                        img_ = torch.load(va_folder + img_name)
                        img.append(img_)
                    img = torch.stack(img).cuda()
                    with torch.no_grad():
                        model.eval()
                        out = model(img)
                        tester.addtable((out[index_predL], out[index_predS]), indices)
                if epoch <= 2: print('start getacc')
                acc = tester.getacc(savefolder + 'tester_train%s_epoch%s_seed%s'%(str(train_n).zfill(2), str(epoch).zfill(5), str(seed).zfill(2)))
                if epoch <= 2: print('end getacc')
                logger.write('epoch=%d loss_te=%s'%(epoch, str(acc)) )
                
                if epoch <= 2: print('start googlesheet')
                googlesheet(epoch, loss, va_loss, str(acc[0]), str(acc[1]))
                if epoch <= 2: print('end googlesheet')
                
                if ex_img:
                    if epoch <2: print('start ex img')
                    
                    # covered_ex = [covered_te, covered_va, covered_te] #no need
                    savenames = ['training_ep', 'validation_ep', 'testing_ep']
                    modenames = ['TR', 'VA', 'TE']
                    for i in range(3):
                        savename = savenames[i]
                        modename = modenames[i]
                        gts_fol_ = ex_gts_fol[i]
                        gtl_fol_ = ex_gtl_fol[i]
                        # covered_ex_ = covered_ex[i] #no need
                        ex_folder_ = ex_folder[i]
                        for _,__,ex_set in os.walk(ex_folder_):
                            print('fin ex_set')

                        va_loss = []
                        permutation = [i for i in range(len(ex_set))]
                        for i in range(0,len(ex_set), batch_size):
                            ii = i+batch_size
                            indices = permutation[i:ii]
                            img = []
                            gts = []
                            gtl = []
                            covered = []
                            for ind in indices: 
                                img_name = ex_set[ind]
                                namei = img_name[:10]
                                # nameint = int(namei) #no need

                                img_ = torch.load(ex_folder_ + img_name)
                                gts_ = torch.load(gts_fol_ + namei + '_2p')
                                gtl_ = torch.load(gtl_fol_ + namei + '_2p')
                                img.append(img_)
                                gts.append(gts_)
                                gtl.append(gtl_)
                                # covered.append(covered_ex_[nameint]) #no need
                            img = torch.stack(img).cuda()
                            gts = torch.stack(gts).cuda()
                            gtl = torch.stack(gtl).cuda()
                            with torch.no_grad():
                                model.eval()
                                out = model(img)
                            savename = savename+str(epoch).zfill(5)+'_seed'+str(seed).zfill(2)
                            header_msg = modename + '_epoch %d'%epoch
                            torch2img.genimg_(img, (out[index_predL],out[index_predS]), gts, gtl, savename, header_msg)
                            msg = '\nimg|img+gts_p|gts|gts_p|gtl|gtl_p\n'
                            msg += ex_folder_[:-1] + ' epoch' +str(epoch)
                            line.send_img('temp.jpg', msg)
                   

            if epoch % saveEvery == 0 or epoch-startepoch == 1  : 
                if epoch <2: print('start save')
                times.append(t.time()-time_)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'time': sum(times)
                    }, savefolder+'train%s_epoch%s_seed%s.pth'%(str(train_n).zfill(2), str(epoch).zfill(10), seed))

                msg = 'epoch=%d saved, used_time=%dmins'%(epoch, sum(times)/60)
                logger.write(msg)
                line.send(msg)
                time_ = t.time()
            
            epoch += 1
            if epoch <2: print('end epoch')
            print(epoch)
        print('fin seed%d'%seed)
    line.send('fin all')    

if __name__ == "__main__":
    run(1)