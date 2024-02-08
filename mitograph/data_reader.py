#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:35:36 2024

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
#%% Make a dictionary with the data
fld_path = "/Users/amansharma/Documents/Data/Mitochondria_masked/"
volume_file = "/Users/amansharma/Documents/Data/Mitochondria_masked/sizes.csv"

try:
    # Read the CSV file into a DataFrame
    volume_array = pd.read_csv(volume_file)
    
except FileNotFoundError:
    print(f"The file '{volume_file}' could not be found.")
except Exception as e:
    print(f"An error occurred: {e}")
cell_wise_dict =[]

for i in range(1,1269):
    
    #keys = ['no. of nodes','degree of nodes','average degree of nodes','volume','length','number of connected components','component properties','mother volume','bud volume']
    
    dction = {} 
    
    
    
    mitograph_path = join(fld_path,str(i)+"/"+str(i)+".mitograph")  # Replace with the path to your text file
    nodes_path = join(fld_path,str(i)+"/"+str(i)+".gnet")
    comp_wise_path = join(fld_path,str(i)+"/"+str(i)+".cc")
    
    try:
        with open(nodes_path, 'r') as file:
            # Read the file line by line
            rows1 = [line.split() for line in file]
        
        degrees = np.zeros((int(rows1[0][0]),1))
                
        #count the degree of each node
        for row_n,row in enumerate(rows1[1:]):
                node_num = np.array(row).astype(float)
                degrees[int(node_num[0])] += 1
                degrees[int(node_num[1])] += 1
        avg = np.average(degrees)
            
        with open(mitograph_path, 'r') as file:
            # Read the file line by line
            rows = [line.split() for line in file]    
            
        with open(comp_wise_path, 'r') as file:
            # Read the file line by line
            rows2 = [line.split() for line in file]    
            
        
        
        
        
        
        
            
        dction['degree of nodes'] = degrees
        dction['average degree of nodes'] = avg
        dction['no. of nodes'] = rows[6][5]
        dction['volume'] = rows[6][0]
        dction['length'] = rows[6][3]
        dction['number of connected components'] = rows[6][7]    
        dction['mother volume'] = volume_array.loc[i,'MotherVol(um3)']
        dction['bud volume'] = volume_array.loc[i,'BudVol(um3)']
            
        cell_wise_dict.append(dction)
            
    except FileNotFoundError:
        print(f"The file '{mitograph_path}' could not be found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    


print(len(cell_wise_dict))
