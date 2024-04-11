#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:02:44 2024

@author: amansharma
"""


#%% libraries
from tifffile import imwrite
from tifffile import imread

import shutil
import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt


from os import walk
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

import networkx as nx

import json

#%% Loading data into arrays
csv_path_gluc = "/Users/amansharma/Documents/Data/Saransh_mito_data/calibration/gluc" 
csv_path_gal = "/Users/amansharma/Documents/Data/Saransh_mito_data/calibration/gal" 

sr_Glu_Nprop = pd.read_csv(os.path.join(csv_path_gluc,"Sr.csv"))
mito_Glu_Nprop = pd.read_csv(os.path.join(csv_path_gluc,"mitograph.csv"))
am_Glu_Nprop = pd.read_csv(os.path.join(csv_path_gluc,"Am.csv"))

sr_Gal_Nprop = pd.read_csv(os.path.join(csv_path_gal,"Sr.csv"))
mito_Gal_Nprop = pd.read_csv(os.path.join(csv_path_gal,"mitograph.csv"))
am_Gal_Nprop = pd.read_csv(os.path.join(csv_path_gal,"Am.csv"))


glu_cID = np.array(sr_Glu_Nprop.loc[:,'cell_id'])
gal_cID = np.array(sr_Gal_Nprop.loc[:,'cell_id'])

sr_gal_T = np.array(sr_Gal_Nprop.loc[:,'T'])
am_gal_T =np.array(am_Gal_Nprop.loc[:,'T'])
mito_gal_T =np.array(mito_Gal_Nprop.loc[:,'T'])
sr_glu_T = np.array(sr_Glu_Nprop.loc[:,'T'])
am_glu_T =np.array(am_Glu_Nprop.loc[:,'T'])
mito_glu_T =np.array(mito_Glu_Nprop.loc[:,'T'])


sr_gal_L = np.array(sr_Gal_Nprop.loc[:,'L'])
am_gal_L =np.array(am_Gal_Nprop.loc[:,'L'])
mito_gal_L =np.array(mito_Gal_Nprop.loc[:,'L'])
sr_glu_L = np.array(sr_Glu_Nprop.loc[:,'L'])
am_glu_L =np.array(am_Glu_Nprop.loc[:,'L'])
mito_glu_L =np.array(mito_Glu_Nprop.loc[:,'L'])



sr_gal_cc = np.array(sr_Gal_Nprop.loc[:,'CC'])
am_gal_cc =np.array(am_Gal_Nprop.loc[:,'CC'])
mito_gal_cc =np.array(mito_Gal_Nprop.loc[:,'CC'])
sr_glu_cc = np.array(sr_Glu_Nprop.loc[:,'CC'])
am_glu_cc =np.array(am_Glu_Nprop.loc[:,'CC'])
mito_glu_cc =np.array(mito_Glu_Nprop.loc[:,'CC'])



sr_gal_J = np.array(sr_Gal_Nprop.loc[:,'J'])
am_gal_J =np.array(am_Gal_Nprop.loc[:,'J'])
mito_gal_J =np.array(mito_Gal_Nprop.loc[:,'J'])
sr_glu_J = np.array(sr_Glu_Nprop.loc[:,'J'])
am_glu_J =np.array(am_Glu_Nprop.loc[:,'J'])
mito_glu_J =np.array(mito_Glu_Nprop.loc[:,'J'])


#%% plotting

fig, axes = plt.subplots()

xl= yl =10

plt.xlim(-0,xl)
plt.ylim(-0,yl)

#Manual comparison - T
plt.scatter(sr_gal_cc,mito_gal_cc,c='r',label='Gal',s=15,marker="v")
plt.scatter(sr_glu_cc,mito_glu_cc,c='g',label='Glu',s=15,marker="^")
plt.plot(range(xl),range(yl),'k--')


plt.title('Manual comparison for number of connected components')
plt.xlabel('Saransh')
plt.ylabel('Mitograph')    
plt.legend()
plt.savefig("/Users/amansharma/Documents/Data/Saransh_mito_data/calibration/SrMito_cc.png",dpi=600)
