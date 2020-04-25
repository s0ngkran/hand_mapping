import pygame as g 
from tkinter import filedialog as fd
import os 
import time
import numpy as np

savefolder = 'save/'
color = [g.Color(255,0,0),g.Color(0,255,0),g.Color(0,0,255)]
print('start program')
filename = fd.askopenfilename()
folder_ = filename.split('/')[:-1]
folder = '/'.join(folder_)+'/'
for _,__,img_names in os.walk(folder):
    print('fin img_names')
for i in range(len(img_names)):
    if img_names[i] == filename.split('/')[-1]:
        imgi = i
        break

img = g.image.load(folder+img_names[imgi])
_,__,width,height = img.get_rect()
scale = 1.2
img = g.transform.scale(img, (int(width*scale), int(height*scale)))

g.init() 
scr = g.display.set_mode((1000,600))
g.display.set_caption('test')
myfont = g.font.SysFont('Comic Sans MS', 20)
running = True
point1 = 0
point2 = 0
rec = []
text = 'Rclick > draw, Lclick > back, Mclick > del all, space > save and next, 7 > save emt, ESC > exit'
while running:
    for event in g.event.get():
        if event.type == g.QUIT:
            running = False
    key = g.key.get_pressed()
    Rclick, Mclick, Lclick = g.mouse.get_pressed()
    if key[g.K_ESCAPE] == 1:
        running = False
    elif Rclick == 1 and point1 == 1:
        point2 = 1
    elif Rclick == 1 and point1 == 0 :
        point1 = 1
        x1, y1 = g.mouse.get_pos()
        time.sleep(0.2)
    elif Lclick == 1 and point1 == 0 and point2==0 and rec != []:
        rec.remove(rec[-1])
        time.sleep(0.5)
    elif Lclick == 1 and (point1==1 or point2 == 1):
        point1 = 0
        point2 = 0
        time.sleep(0.5)
    elif Mclick == 1:
        point1 = 0
        point2 = 0
        rec = []
    elif key[g.K_SPACE] == 1 and rec != []:
        np.save(savefolder+'peaks_'+img_names[imgi],rec)
        imgi += 1
        img = g.image.load(folder+img_names[imgi])
        _,__,width,height = img.get_rect()
        img = g.transform.scale(img, (int(width*scale), int(height*scale)))
        
        point1 = 0
        point2 = 0
        rec = []
        time.sleep(0.5)
    elif key[g.K_7] == 1 :
        rec = ['blank']
        np.save(savefolder+'peaks_'+img_names[imgi],rec)
        imgi += 1
        img = g.image.load(folder+img_names[imgi])
        _,__,width,height = img.get_rect()
        img = g.transform.scale(img, (int(width*scale), int(height*scale)))
        
        point1 = 0
        point2 = 0
        rec = []
        time.sleep(0.5)



    scr.fill((0,0,10))
    scr.blit(img, (0,0))
    if point1 == 1:
        if point2 == 0:
            x2, y2 = g.mouse.get_pos()
        g.draw.polygon(scr, color[0], [(x1,y1),(x1,y2),(x2,y2),(x2,y1)], 1)
        center = int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)
        g.draw.circle(scr,color[0],center,3)
    if point1 == 1 and point2 == 1:
        imgx = x1+(x2-x1)/2
        imgy = y1+(y2-y1)/2 
        imgw = x2-x1 
        imgh = y2-y1
        imgx /= width 
        imgy /= height
        imgw /= width 
        imgh /= height

        x_, y_ = g.mouse.get_pos()
        showtex = myfont.render('press 1 or 2 !!!', False, (255, 0, 255))
        scr.blit(showtex,(x_,y_))
        if key[g.K_1] == 1:
            point1 = 0
            point2 = 0
            rec.append([imgx,imgy,imgw,imgh,1,1,0])
        elif key[g.K_2] == 1:
            point1 = 0
            point2 = 0
            rec.append([imgx,imgy,imgw,imgh,1,0,1])

    if rec == []:
        pass 
    else:
        for obj in rec:
            x = obj[0]*width
            y = obj[1]*height
            w = obj[2]*width
            h = obj[3]*height
            class1 = obj[5]
            if class1 == 1:
                col = color[1]
            else :
                col = color[2]
            g.draw.circle(scr,col,(int(x),int(y)),3)
            x1_ = int(x-w/2)
            x2_ = int(x+w/2 )
            y1_ = int(y-h/2 )
            y2_ = int(y+h/2)
            g.draw.polygon(scr, col, [(x1_,y1_),(x1_,y2_),(x2_,y2_),(x2_,y1_)], 1)
    textsurface = myfont.render(text, False, (0, 255, 255))
    scr.blit(textsurface,(10,500))
    g.display.update()
    

