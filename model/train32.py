import os
import cv2 
import torch
import numpy as np
import time
import torch.optim as optim
from Logger import Logger
import time
from torch2img import torch2img as t2img
from torchvision.utils import save_image
class dataloader:
    def __init__(self, imgs_file, gts_file, gtl_file):
        assert imgs_file[-6:] == gts_file[-6:] == gtl_file[-6:] =='.torch'
        print('-----------------------')
        print('loading...')
        print(imgs_file)
        print(gts_file)
        print(gtl_file)
        print('')
        self.imgs = torch.load(imgs_file)
        print('loaded img')
        self.gts = torch.load(gts_file)
        print('loaded gts')
        self.gtl = torch.load(gtl_file)
        print('loaded gtl')
        self.size = len(self.imgs)-1
        print('size = ', self.size)
    def getsmall(self, lastind):
        size = lastind
        self.imgs = self.imgs[:size+1]
        self.gts = self.gts[:size+1]
        self.gtl = self.gtl[:size+1]
        self.size = size
        print('getsmall size=',size)
        
class Trainer:
    def __init__(self, model, channel, lr, lossfunction, thisname, savefolder):
        assert savefolder[-1] == '/'
        assert channel in [1,3]
        train_n = thisname.split('train')[1].split('.py')[0]
        self.thisname = thisname
        self.model = model().cuda()
        self.lr_ori = lr
        self.optimizer = optim.Adam(self.model.parameters())
        logfile = 'tr%s'%(train_n)
        self.logname = logfile
        self.logger = Logger(logfile)
        self.savefolder = savefolder
        self.start_epoch = 0
        self.epoch = 0
        self.train_n = train_n
        self.times = 0
        self.lossfunction = lossfunction
        self.torch2img = t2img(thisname)
        self.unpack_func = self.to_grayscale if channel==1 else self.to_3channel
        
    def loadstage(self, stagefile):
        assert stagefile[-4:] =='.pth'
        checkpoint = torch.load(stagefile)
        self.model.load_state_dict(checkpoint['model_state_dict'])           #
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   #
        self.start_epoch = checkpoint['epoch']+1                              #
        self.times = checkpoint['time']                                   #
        print('fin import state')                                       #
        self.logger.write('loaded model_stage')
    
    def setdata(self, training_set, validation_set, testing_set=None):
        self.training_set = training_set
        self.validation_set = validation_set
        if testing_set != None:
            testing_set.getsmall(5) ############## save mem ##########################
            self.testing_set = testing_set

    def addtime(self):
        self.times += time.time()-self.t0
        self.t0 = time.time()
    def savemodel(self, ):
        self.addtime()
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'time': self.times,
            }, self.savefolder+'train%s_epoch%s_seed%s.pth'%(str(self.train_n).zfill(2), str(self.epoch).zfill(10), self.seed))

        msg = 'epoch=%d saved, used_time=%dmins'%(self.epoch, self.times/60)
        self.logger.write(msg)
        self.t0 = time.time()
    def scale_lr(self):
        n_data = self.training_set.size
        all_iter = n_data//self.batch
        lr = self.lr_ori / all_iter
        self.change_lr(lr)
    def change_lr(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.logger.write('learning rate to' + str(lr))
    def init_training(self):
        self.scale_lr()
        self.n_data_tr = self.training_set.size
        self.epoch = self.start_epoch

        indices = torch.randperm(self.training_set.size) + 1
        self.ex_training_ind = indices[:self.batch]
        indices = torch.randperm(self.validation_set.size) + 1
        self.ex_validation_ind = indices[:self.batch]
        indices = torch.randperm(self.testing_set.size) + 1
        self.ex_testing_ind = indices[:self.batch]
        self.logger.write('tr_size'+str(self.training_set.size))
        self.logger.write('va_size'+str(self.validation_set.size))
        self.logger.write('te_size'+str(self.testing_set.size))

    def train(self):
        self.model.train()
        time0 = time.time()
        iteration = 0
        toggle = True
        self.permutation = torch.randperm(self.n_data_tr) +1
        for i in range(0, self.n_data_tr, batch):
            iteration += 1
            self.optimizer.zero_grad()
            ii = i+batch
            if ii > self.n_data_tr:break
            indices = self.permutation[i:ii]

            img, gts, gtl = self.unpack(self.training_set, indices)

            output = self.model(img)
            self.loss = self.lossfunction(output, gtl, gts)
            self.loss.backward()
            self.optimizer.step()

            if toggle:
                toggle = False
                self.logger.write('first loss %.5f'%self.loss)

            if time.time() - time0 > 60:
                time0 = time.time()
                msg = 'epoch_=%d iter_=%d loss_=%.5f'%(self.epoch, iteration, self.loss)
                self.logger.write(msg)

            if iteration > 3 and self.MODE == 'test_mode':
                break
        self.logger.write('epoch=%d loss=%.5f'%(self.epoch, self.loss))

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            va_loss = []
            n_data = self.validation_set.size
            ind = [i+1 for i in range(n_data)]
            iteration = 0
            for i in range(0, n_data, self.batch):
                iteration += 1
                ii = i+self.batch
                if ii > n_data: ii = n_data
                indices = ind[i:ii]

                img, gts, gtl = self.unpack(self.validation_set, indices)
                output = self.model(img)
                va_loss_ = self.lossfunction(output, gtl, gts)
                va_loss.append(va_loss_)
                if iteration>3 and self.MODE == 'test_mode':
                    break
            va_loss = sum(va_loss)/len(va_loss)
            self.logger.write('epoch=%d loss_va=%.5f'%(self.epoch, va_loss) )
    def ex_img(self, data_set, ind, modename):
        ############# config ######################
        index_predL, index_predS = 2, 3
        #######################################
        self.model.eval()
        with torch.no_grad():
            n_data = self.batch

            img, gts, gtl = self.unpack(data_set, ind)

            out = self.model(img)
            out = (out[index_predL], out[index_predS])
        savename = self.thisname+'_'+modename+'_ep' + str(self.epoch).zfill(5)+'_seed'+str(self.seed).zfill(2)
        header_msg = modename + '_epoch %d'%self.epoch
        self.torch2img.genimg_(img=img, out=out, gts=gts, gtl=gtl, filename=savename, msg=header_msg, savefolder=None)

    def to_grayscale(self, bgr_img):
        # rgb_weight = [0.2989, 0.5870, 0.1140]
        img = self.to_3channel(bgr_img)
        bgr_weight = torch.FloatTensor([0.1140, 0.5870, 0.2989])
        img = torch.matmul(img.transpose(2,0), bgr_weight).unsqueeze(0)
        return img
    def to_3channel(self, img):
        return img.to(dtype=torch.float32)/255
        
    def unpack(self, data_set, indices):
        img = torch.stack([self.unpack_func(data_set.imgs[i]) for i in indices]).cuda()
        gts = torch.stack([data_set.gts[i] for i in indices]).cuda()
        gtl = torch.stack([data_set.gtl[i] for i in indices]).cuda()
        return img, gts, gtl
    
    def test_run(self, batch):
        self.logger.write('''
        ################
        start test_run
        ################''')
        self.MODE = 'test_mode'
        to_epoch = 2
        self.batch = batch
        self.init_training()
        #############
        self.n_data_tr = self.batch
        ###########
        self.seed = 1
        self.t0 = time.time()
        self.logger.write('start training, test_mode')
        while self.epoch < to_epoch: # start epoch = 0
            self.epoch += 1
            self.logger.write('start train')
            self.train()

            self.logger.write('start va')
            self.validation()

            self.logger.write('start ex_img')
            self.ex_img(self.training_set, self.ex_training_ind, 'TR')
            self.ex_img(self.validation_set, self.ex_validation_ind, 'VA')
            self.ex_img(self.testing_set, self.ex_testing_ind, 'TE')
            self.logger.write('fin epoch'+str(self.epoch))
            self.savemodel()

    def run(self, batch, to_epoch, va_every, seed):

        self.MODE = 'train_mode'
        self.batch = batch
        self.init_training()
        self.seed = seed
        self.t0 = time.time()
        self.logger.write('''
        ##############
        start training
        ##############''')
        while self.epoch < to_epoch: # start epoch = 0
            time0 = time.time()
            self.epoch += 1
            self.train()
            time_ = time.time() - time0
            self.logger.write('time tr '+str(time_/60)+'mins')
            
            time0 = time.time()
            if self.epoch % va_every == 0:
                self.validation()
                self.ex_img(self.training_set, self.ex_training_ind, 'TR')
                self.ex_img(self.validation_set, self.ex_validation_ind, 'VA')
                self.ex_img(self.testing_set, self.ex_testing_ind, 'TE')
                self.savemodel()
            time_ = time.time() - time0
            self.logger.write('time va and ex_img '+str(time_/60)+'mins\n')

    def test_out(self, weight_file):
        
        assert weight_file[-4:] == '.pth'
        ##################################
        savefolder = 'temp/'

        ################################
        self.loadstage(weight_file)
        self.epoch = self.start_epoch-1 

        indices = torch.randperm(self.training_set.size) + 1
        self.ex_training_ind = indices[:2] # for dummy stack
        print( 'img_ind = ', self.ex_training_ind[0])
        self.model.eval()
        with torch.no_grad():
            img, gts, gtl = self.unpack(self.training_set, self.ex_training_ind)
            out1, out2, out3, out4 = self.model(img) 
            
            # batch, part, x, y
            out1 = out1[0].mean(0)
            out2 = out2[0].mean(0)
            out3 = out3[0].mean(0)
            out4 = out4[0].max(0)[0]

            self.save_img(out1, savefolder+'out1')
            self.save_img(out2, savefolder+'out2')
            self.save_img(out3, savefolder+'out3')
            self.save_img(out4, savefolder+'out4')

    def save_img(self, tensor, savefile):
        assert len(tensor.shape) == 2
        save_image(tensor.transpose(0,1), savefile+'.bmp')
        print('saved', savefile+'.bmp')


if __name__ == "__main__":
    from hand_model_c3_dropout import hand_model as model
    from loss_function_weight import loss_func as lossfunction
 
    thisname = os.path.basename(__file__)
    
    print('starting...', thisname)
    ################### config ##############################################
    savefolder = 'save/'
    channel = 3
    lr = 0.02
    batch = 5
    to_epoch = 3000
    va_every = 1
    seed = 1

    img = '../data014/training/imgs_replaced_background.torch' 
    gts = '../data014/training/gts_replaced_background.torch'
    gtl = '../data014/training/gtl_replaced_background.torch'
    training_set = dataloader(img,gts,gtl)

    img = '../data014/validation/imgs.torch'
    gts = '../data014/validation/gts.torch'
    gtl = '../data014/validation/gtl.torch'
    validation_set = dataloader(img,gts,gtl)

    img = '../data014/testing/imgs.torch'
    gts = '../data014/testing/gts.torch'
    gtl = '../data014/testing/gtl.torch'
    testing_set = dataloader(img,gts,gtl)
    ################################################################################
    trainer = Trainer( model, channel, lr, lossfunction, thisname, savefolder)

    # training_set.getsmall()

    trainer.setdata(training_set, validation_set, testing_set)
    del training_set, validation_set, testing_set
    

    trainer.test_run(batch = 5)
    trainer.run( batch, to_epoch, va_every, seed)
    # weight = 'save/train20_epoch0000000001_seed1.pth'
    # trainer.test_out(weight)