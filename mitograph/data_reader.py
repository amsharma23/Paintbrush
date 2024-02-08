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

import json
#%% Make a dictionary with the data
fld_path = "/Users/amansharma/Documents/Data/Mitochondria_masked/"
volume_file = "/Users/amansharma/Documents/Data/Mitochondria_masked/sizes.csv"

try:
    # Read the CSV file into a DataFrame
    volume_array = pd.read_csv(volume_file)
    print(volume_array)
except FileNotFoundError:
    print(f"The file '{volume_file}' could not be found.")
except Exception as e:
    print(f"An error occurred: {e}")
cell_wise_dict =[]

df = pd.DataFrame(columns=['no. of nodes','degree of nodes','average degree of nodes','volume','length','number of connected components','component properties','mother volume','bud volume','no. of loops'])


for i in range(1,1269):
    
    #keys = ['no. of nodes','degree of nodes','average degree of nodes','volume','length','number of connected components','component properties','mother volume','bud volume','no. of loops']
    
    dction = {} 
    
    
    
    mitograph_path = join(fld_path,str(i)+"/"+str(i)+".mitograph")  # Replace with the path to your text file
    nodes_path = join(fld_path,str(i)+"/"+str(i)+".gnet")
    #comp_wise_path = join(fld_path,str(i)+"/"+str(i)+".cc")
    
    try:
        with open(nodes_path, 'r') as file:
            # Read the file line by line
            rows1 = [line.split() for line in file]
        
        degrees = np.zeros((int(rows1[0][0]),1))
        loops =0        
        #count the degree of each node
        for row_n,row in enumerate(rows1[1:]):
                node_num = np.array(row).astype(float)
                if(node_num[0]!=node_num[1]): #for loops the same node connected to itself 
                    degrees[int(node_num[0])] += 1
                    degrees[int(node_num[1])] += 1
                else:
                    loops+=1
                
        avg = np.average(degrees)
            
        with open(mitograph_path, 'r') as file:
            # Read the file line by line
            rows = [line.split() for line in file]
            
        
        #component wise volume, number of nodes and avergae degree of nodes    
        num_comps = int(rows[6][7])
        comp_props = []     
        
        for i in range(num_comps):
            # volume of component
            #number of nodes
            l = [rows[11+i][3],rows[11+i][0]]
             
            
            comp_props.append(l)
        
        
        
        
        #dction['component properties'] = comp_props    #
        #dction['degree of nodes'] = degrees  #
        dction['no. of loops'] = loops
        dction['average degree of nodes'] = avg
        dction['no. of nodes'] = rows[6][5]
        dction['volume'] = rows[6][0]
        dction['length'] = rows[6][3]
        dction['number of connected components'] = int(num_comps)    
        dction['mother volume'] = volume_array.loc[i,'MotherVol(um3)']
        dction['bud volume'] = volume_array.loc[i,'BudVol(um3)']
        output= pd.DataFrame([dction])
        cell_wise_dict.append(dction)
        df = pd.concat([df, output], ignore_index=True)
        df.to_csv('sizes_analyzed.csv', encoding='utf-8')
    except FileNotFoundError:
        print(f"The file '{mitograph_path}' could not be found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    


# saving json solution from: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

save_path = "/Users/amansharma/Documents/Data/Mitochondria_masked/"+"Cell_wise_Network_prop.json"

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

dumped = json.dumps(cell_wise_dict, cls=NumpyEncoder)


with open(save_path,'w') as json_file:
    json.dump(dumped, json_file)


#%% JSON to 


#output= pd.DataFrame([{ ‘MotherVol’:mother_size,‘BudVol’:bud_size}])
#df = pd.concat([df, output], ignore_index=True)
#df.to_csv(str(“sizes”)+“.csv”, encoding=‘utf-8’)