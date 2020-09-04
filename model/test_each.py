import torch 
import torch.optim as optim
import time as t
import os
from torch2img import torch2img


from hand_model_dropout import hand_model
indexPredL, indexPredS = 2, 3
from test_function_1block import Tester


class Tester_img:
    def __init__(self, weight, img_folder, gts_folder, gtl_folder, savename, savefolder
                    , indexPredL=indexPredL, indexPredS=indexPredS, suffix_gt='_2p'):
        assert img_folder[-1] == gts_folder[-1] == gtl_folder[-1] == savefolder[-1] == '/'
        print('init tester')
        self.indL, self.indS = indexPredL, indexPredS
        self.savename = savename
        self.savefolder = savefolder
    
        self.model = hand_model().cuda()
        self.optimizer = optim.Adam(self.model.parameters())

        checkpoint = torch.load(weight)
        self.model.load_state_dict(checkpoint['model_state_dict'])           #
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   #
        self.epoch = checkpoint['epoch']                                     #
        self.times = [checkpoint['time']]                                    #
        print('loaded model\'s stage')

        self.img_folder = img_folder
        for _,__,img_names in os.walk(img_folder):
            print('fin walk')
        self.img_names = [self.img_folder+img_name for img_name in img_names]
        assert int(img_names[0][:10]) > 0 
        namei = [img_name[:10] for img_name in img_names]
        self.namei = namei
        self.gts_names = [gts_folder+name+suffix_gt for name in namei]
        self.gtl_names = [gtl_folder+name+suffix_gt for name in namei]

        self.genimg = torch2img(savename) # init torch2img class
    def feed(self, img):
        self.model.eval()
        with torch.no_grad():
            out = self.model(img)
        return out[self.indL], out[self.indS]
    def test(self):
        lst = zip(self.img_names, self.gts_names, self.gtl_names, self.namei)
        for img_name, gts_name, gtl_name, namei in lst:
            print(namei)
            img = torch.load(img_name).unsqueeze(0).cuda()
            gts = torch.load(gts_name).unsqueeze(0).cuda()
            gtl = torch.load(gtl_name).unsqueeze(0).cuda()
            filename = namei
           
            header_msg = 'ep' + str(self.epoch) + '_' + str(int(namei))
            out = self.feed(img)
            self.genimg.genimg_(img, out, gts, gtl, filename, msg = header_msg, savefolder=self.savefolder)
if __name__ == "__main__":
    print('start')
    weight = 'save/train03_epoch0000000100_seed1.pth'
    img_folder = '../data014/testing/img_torch/'
    gts_folder = '../data014/testing/gts/'
    gtl_folder = '../data014/testing/gtl/'
    savename = 'test_tr3'
    savefolder = 'test_each/'
    
    tester = Tester_img(weight, img_folder, gts_folder, gtl_folder, savename,  savefolder)
    tester.test()
    print('finish')
