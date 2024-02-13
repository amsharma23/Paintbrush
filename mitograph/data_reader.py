#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:35:36 2024

@author: amansharma
Network Analysis of mitochondrial images 
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

df = pd.DataFrame(columns=['no. of nodes','degree of nodes','average degree of nodes','volume','length','number of connected components','component properties','MotherVol(um3)','BudVol(um3)','no. of loops','label'])


for j in range(1,1269):
    
    #keys = ['no. of nodes','degree of nodes','average degree of nodes','volume','length','number of connected components','component properties','mother volume','bud volume','no. of loops']
    
    dction = {} 
    
    
   
    mitograph_path = join(fld_path,str(j)+"/"+str(j)+".mitograph")  # Replace with the path to your text file
    nodes_path = join(fld_path,str(j)+"/"+str(j)+".gnet")
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
        dction['MotherVol(um3)'] = volume_array.loc[j,'MotherVol(um3)']
        dction['BudVol(um3)'] = volume_array.loc[j,'BudVol(um3)']
        #print(j)
        dction['label'] = str(j)
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


#%% Stats and plotting


an_csv_path  = "/Users/amansharma/Documents/Data/Mitochondria_masked/sizes_analyzed.csv"
volume_array_array = pd.read_csv(an_csv_path)

bud_vols = volume_array.loc[:,'BudVol(um3)']
bins = np.linspace(min(bud_vols), max(bud_vols),10)
binned_no_loops, bin_edges, binnumber = sp.stats.binned_statistic(bud_vols, volume_array.loc[:,'no. of loops'],statistic='median', bins=10)

binned_deg, bin_edges, binnumber = sp.stats.binned_statistic(bud_vols, volume_array.loc[:,'average degree of nodes'],statistic='median', bins=10)
binned_nodes, bin_edges, binnumber = sp.stats.binned_statistic(bud_vols, volume_array.loc[:,'no. of nodes'],statistic='median', bins=10)
binned_mito_vol, bin_edges, binnumber = sp.stats.binned_statistic(bud_vols, volume_array.loc[:,'volume'],statistic='median', bins=10)
binned_mito_length, bin_edges, binnumber = sp.stats.binned_statistic(bud_vols, volume_array.loc[:,'length'],statistic='median', bins=10)
binned_no_cc, bin_edges, binnumber = sp.stats.binned_statistic(bud_vols, volume_array.loc[:,'number of connected components'],statistic='median', bins=10)



print(bins,binned_mito_vol)
plt.figure()
plt.hlines(binned_mito_vol,bin_edges[:-1], bin_edges[1:],label='Mito Volume(um3)',color='g')
plt.hlines(binned_deg, bin_edges[:-1], bin_edges[1:], label='Avg degree',color='r')
plt.hlines(binned_nodes, bin_edges[:-1], bin_edges[1:], label='number of nodes',color='b')
plt.xlabel('BudVol(um3)')
plt.legend()

#%% Montage making
an_csv_path  = "/Users/amansharma/Downloads/mitograph_data.csv"
if not os.path.exists("/Users/amansharma/Documents/Data/Mitochondria_masked/MaxProj_sorted/"):
    os.mkdir("/Users/amansharma/Documents/Data/Mitochondria_masked/MaxProj_sorted/")
props_arr = pd.read_csv(an_csv_path)
bud_vols = props_arr.loc[:,'BudVol(um3)']
bud_vols_sort = np.sort(bud_vols)
labels = np.array(props_arr.loc[:,'label'])

labels_sort = []
i_p = -1
for i in bud_vols_sort:
    
    if not (i_p == i or i==0):
        
        mathced_ind = [j_n for j_n,j in enumerate(bud_vols) if j==i ]
        labels_sort.append(labels[mathced_ind])
        i_p =i
        
#print(labels_sort)
count=0        
for i_n,i in enumerate(labels_sort):
    for j in i:
        shutil.copyfile("/Users/amansharma/Documents/Data/Mitochondria_masked/"+str(j)+"/"+str(j)+".png",f"/Users/amansharma/Documents/Data/Mitochondria_masked/MaxProj_sorted/"+str(count)+".png")
        if(count==326):
            print(j,count)
        count+=1


#%% Network X analysis

G = nx.MultiGraph()
net_arr = []
net_path =  "/Users/amansharma/Documents/Data/Saransh_mito_data/Mitochondria_masked_glu/"
for i in range(1,1270):
    file_path =  os.path.join(net_path,str(i)+"/"+str(i)+".gnet")
    with open(file_path,'r') as net_file:
        
        nodes = net_file.readlines()
        nodes = [str(nd.split('\t')[0])+" "+str(nd.split('\t')[1]) for nd in nodes[1:]]
        #print((nodes))
        G = nx.parse_adjlist(nodes, nodetype=int)
        net_arr.append(G)
        try:
            cyc = nx.find_cycle(G)                
            print('Cycle size for '+str(i)+' is '+str(len(cyc))+'\n')
            
           
                
        except nx.exception.NetworkXNoCycle:
            print('No cycle for '+str(i)+'\n')

