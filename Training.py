#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:04:16 2020

@author: brad
"""
# prerequisites
import torch
import numpy as np
from sklearn import svm
from torchvision import datasets, transforms
from torchvision.utils import save_image


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
import config 

config.init()

from config import  numcolors, args

from mVAE import *

#define color labels 
#this list of colors is randomly generated at the start of each epoch (down below)

#numcolors indicates where we are in the color sequence 
#this index value is reset back to 0 at the start of each epoch (down below)

#this is the amount of variability in r,g,b color values (+/- this amount from the baseline)

#these define the R,G,B color values for each of the 10 colors.  
#Values near the boundaries of 0 and 1 are specified with colorrange to avoid the clipping the random values
    



for epoch in range(1, 201):

    numcolors = 0
    train(epoch, 'iterated')
      # regenerate the list of color labels at the start of each test epoch
    numcolors = 0
    if epoch % 5 == 0:
        test('noskip')

if epoch in [1, 25,50,75,100,150,200,300,400,500]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{args.dir}/checkpoint_twoloss_singlegrad'+str(epoch)+'.pth')



load_checkpoint('output1/checkpoint_twoloss_singlegrad200.pth')
for epoch in range (1,201):
    global colorlabels, numcolors
    #colorlabels = np.random.randint(0, 10,
                                    #100000)  # regenerate the list of color labels at the start of each test epoch
    
    train_labels (epoch)
   
    #if epoch % 5 ==0:
     #   test_labels()
    if epoch in [1, 25,50,75,100,150,200,300,400,500]:
        checkpoint =  {
                 'state_dict_shape_labels': vae_shape_labels.state_dict(),
                 'state_dict_color_labels': vae_color_labels.state_dict(),

                 'optimizer_shape' : optimizer_shapelabels.state_dict(),
                 'optimizer_color': optimizer_colorlabels.state_dict(),

                      }
        torch.save(checkpoint,f'{args.dir}/checkpoint_shapelabels'+str(epoch)+'.pth')
        torch.save(checkpoint, f'{args.dir}/checkpoint_colorlabels' + str(epoch) + '.pth')
