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
import skimage.io as skio

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
fld_path = "/Users/amansharma/Documents/Data/Saransh_mito_data/Mitochondria_masked_Ethanol/"
#volume_file = "/Users/amansharma/Documents/Data/Mitochondria_masked/sizes.csv"
"""
try:
    # Read the CSV file into a DataFrame
    volume_array = pd.read_csv(volume_file)
    print(volume_array)
except FileNotFoundError:
    print(f"The file '{volume_file}' could not be found.")
except Exception as e:
    print(f"An error occurred: {e}")
"""
cell_wise_dict =[]
df = pd.DataFrame(columns=['no. of nodes','degree of nodes','average degree of nodes','volume','length','number of connected components','component properties','MotherVol(um3)','BudVol(um3)','no. of loops','label'])


for j in range(1,2021):
    
    #keys = ['no. of nodes','degree of nodes','average degree of nodes','volume','length','number of connected components','component properties','mother volume','bud volume','no. of loops']
    
    dction = {} 
    
    
   
    mitograph_path = join(fld_path,str(j)+"/"+str(j)+".mitograph")  # Replace with the path to your text file
    nodes_path = join(fld_path,str(j)+"/"+str(j)+".gnet")
    #the format of gnet: 1st line number of nodes; then node pairs each line with distance between them (ni nj dij)
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
    
        dction['average degree of nodes'] = avg
        dction['no. of nodes'] = rows[6][5]
        dction['volume'] = rows[6][0]
        dction['length'] = rows[6][3]
        dction['number of connected components'] = int(num_comps)    
        #dction['MotherVol(um3)'] = volume_array.loc[j,'MotherVol(um3)']
        #dction['BudVol(um3)'] = volume_array.loc[j,'BudVol(um3)']
        #print(j)
        dction['label'] = str(j)
        output= pd.DataFrame([dction])
        cell_wise_dict.append(dction)
        df = pd.concat([df, output], ignore_index=True)
        df.to_csv(os.path.join(fld_path,'sizes_analyzed_ethanol.csv'), encoding='utf-8')
    except FileNotFoundError:
        print(f"The file '{mitograph_path}' could not be found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    


#%% saving json solution from: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

save_path = os.path.join(fld_path,"Cell_wise_Network_prop.json")

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
    
#%% Network X analysis


net_arr = []
net_path =  "/Users/amansharma/Documents/Data/Saransh_mito_data/Mitochondria_masked_Ethanol/"
for i in range(1,2021):
    G = nx.MultiGraph()
    file_path =  os.path.join(net_path,str(i)+"/"+str(i)+".gnet")
    with open(file_path,'r') as net_file:
        
        nodes = net_file.readlines()
        nodes = [str(nd.split('\t')[0])+" "+str(nd.split('\t')[1]) for nd in nodes[1:]]
        
        edges = [(int(j.split()[0]),int(j.split()[1])) for j in nodes]
        #print(edges)
        G.add_edges_from(edges, nodetype=int)
        net_arr.append(G)
print(len(net_arr))        
#%% Network plotting Gal - 35x36; Eth - 44x46; richGlu - 44x46

for g_n,graph in enumerate(net_arr):
    """
    try:
        cyc = nx.cycle_basis(graph)
        print(cyc)
    except nx.exception.NetworkXNoCycle:
        print("no cycle")
    """
    
   
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos,label=graph.nodes, node_color = 'r', node_size = 50, alpha = 1)
    ax = plt.gca()
    print(len(graph.edges))
    for e in graph.edges:
        ax.annotate("",xy=pos[e[0]], xycoords='data',xytext=pos[e[1]], textcoords='data',
                                    arrowprops=dict(arrowstyle="-", color="black",connectionstyle="arc3,rad=rrr".replace('rrr',str(0.2*(e[2]+1))
                                                                    )))
                    
    plt.axis('off')
    file_path =  os.path.join(net_path,str(g_n+1)+"/"+str(g_n+1)+"_graph.png")
    print(file_path)
    plt.savefig(file_path)
    ax.clear()
    plt.clf()

#%% Degree distribution




#%% Montage making - sorts by bud size and now sort all other relevant info in the order of bud size order
an_csv_path  = "/Users/amansharma/Documents/Data/Saransh_mito_data/Mitochondria_masked_gal/sizes_analyzed_gal.csv"
if not os.path.exists("/Users/amansharma/Documents/Data/Saransh_mito_data/Mitochondria_masked_gal/graphs/"):
    os.mkdir("/Users/amansharma/Documents/Data/Saransh_mito_data/Mitochondria_masked_gal/graphs/")
    
    
props_arr = pd.read_csv(an_csv_path)
bud_vols = props_arr.loc[:,'BudVol(um3)']
bud_vols_sort = np.sort(bud_vols)
#bud_vols_sort = bud_vols_sort.tolist()
labels = np.array(props_arr.loc[:,'label'])
MotherVol = np.array(props_arr.loc[:,'MotherVol(um3)'])

no_cc = np.array(props_arr.loc[:,'number of connected components'])
avg_deg_nodes = np.array(props_arr.loc[:,'average degree of nodes'])
no_nodes = np.array(props_arr.loc[:,'no. of nodes'])


labels_sort = []
moth_vol_sort = []
no_cc_sort = []
avg_deg_nodes_sort = []
no_of_nodes_sort = []
i_p = -1

#label sort by smallest bud vol to largest 
for i in bud_vols_sort:
    
    if not (i_p == i):
        
        mathced_ind = [j_n for j_n,j in enumerate(bud_vols) if j==i ]
        
        for m in mathced_ind:
            labels_sort.append(labels[m])
            moth_vol_sort.append(MotherVol[m])
            no_cc_sort.append(no_cc[m])
            avg_deg_nodes_sort.append(avg_deg_nodes[m])
            no_of_nodes_sort.append(no_nodes[m])
    i_p = i
    

moth_vol_sort_bin, bin_edges, bin_no = stats.binned_statistic(bud_vols_sort,moth_vol_sort,statistic='mean',bins=20)
no_cc_sort_bin, bin_edges, bin_no = stats.binned_statistic(bud_vols_sort,no_cc_sort,statistic='mean',bins=20)
avg_deg_nodes_sort_bin, bin_edges, bin_no = stats.binned_statistic(bud_vols_sort,avg_deg_nodes,statistic='mean',bins=20)        
no_of_nodes_sort_bin, bin_edges, bin_no =stats.binned_statistic(no_of_nodes_sort,moth_vol_sort,statistic='mean',bins=20)

#%% plot binned data

#print((bin_no))
#print(bin_edges)
#print(moth_vol_sort_bin)
print(no_of_nodes_sort_bin)
prps = [moth_vol_sort_bin,no_cc_sort_bin,avg_deg_nodes_sort_bin]
       

plt.figure()
plt.hlines(no_of_nodes_sort_bin , bin_edges[:-1], bin_edges[1:], colors='r', lw=2,
           label='binned statistic of data')
plt.xlim(0,80)
plt.ylim(0,70)
plt.title('Nodes')
plt.xlabel('Bud Vol(um3)')
plt.ylabel('number')    
plt.show()

