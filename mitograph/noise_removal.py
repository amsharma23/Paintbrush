#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:00:59 2023

@author: amansharma
"""

#%% libraries
from tifffile import imwrite
from tifffile import imread

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import walk
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


#%% Function

image1_loc = "/Users/amansharma/Documents/Data/Aman_paintbrush/20231103/Run__6/wait_time_30_Z_steps_120_stepV_0.005/segment/0_mask.tif"
image2_loc = "/Users/amansharma/Documents/Data/Aman_paintbrush/20231103/Run__6/wait_time_30_Z_steps_120_stepV_0.005/"
save_loc = "/Users/amansharma/Documents/Data/Aman_paintbrush/20231103/Run__6/wait_time_30_Z_steps_120_stepV_0.005/segment/UP/"


for i in range(1,121):
    
    img_1 = imread(image1_loc)
    img_1[np.nonzero(img_1)] = 1
    img_2 = imread(image2_loc+str(i)+".tif")
    multi_img = np.multiply(img_1,img_2)
    
    imwrite(save_loc+str(i)+".tif",multi_img)
    
#%% Average finder

for i in range(7,10):
    loc = "/Users/amansharma/Documents/Data/Aman_paintbrush/20231103/Run__"+str(i)+"/wait_time_30_Z_steps_120_stepV_0.005/1.tif"
    tif_st = "/Users/amansharma/Documents/Data/Aman_paintbrush/Piezo_z_stack/20231103/Run__"+str(i)+"/wait_time_30_Z_steps_120_stepV_0.005/DOWN/TIF_stack.tif"
    img = imread(loc)
    avg = np.average(img)
    std = np.std(img)
    
    
    #avg = int(avg)
    print(avg,std)
    imgs = imread(tif_st)
    imgs = np.subtract(imgs,avg)
    print(imgs[1,:,:])
    imgs[imgs<0] = 0
    print(imgs[1,:,:])
    imgs = imgs.astype(np.uint8)
    print(imgs[1,:,:])
    imwrite("/Users/amansharma/Documents/Data/Aman_paintbrush/Piezo_z_stack/20231103/Run__"+str(i)+"/wait_time_30_Z_steps_120_stepV_0.005/DOWN/Noise_rem/TIF_stack_noise_rem.tif",imgs)