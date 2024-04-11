#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Mon Apr 08 2024 : original code - https://github.com/aelefebv/nellie-supplemental/blob/main/case_studies/GNN/graph_frame.py ; https://github.com/aelefebv/nellie/tree/main/nellie

@author: amansharma
"""
#%%
from nellie import logger
from nellie.im_info.im_info import ImInfo
from nellie.utils.general import get_reshaped_image
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import pandas as pd
import math
from tifffile import imwrite
from tifffile import imread
import os
import networkx as nx

#%%

bin_folder = "/Users/amansharma/Documents/Data/Nellie_test/bin_70_80_manual/"
tif_files = os.listdir(bin_folder)
tif_files = [file.removesuffix('.tif') for file in tif_files if file.endswith(".tif")]
tif_files.sort()
print(tif_files)

#%%
df_average = pd.DataFrame(columns=['File#','#CC(Islands)','#Nodes','#Tips(Deg1)','#Junctions(Deg3+)','#Loops','Avg Deg'])
cc , tips, junc, loops = [0,0,0,0]
for n_file,file in enumerate(tif_files):
    
    df_average.loc[n_file,'File#'] = file
    pix_calss_path = os.path.join(bin_folder,"nellie_output",file+"-ch0-im_pixel_class.ome.tif")
    pix_class = imread(pix_calss_path)
    
    df_nodewise = pd.DataFrame(columns=['CC(Island)#','Node#','Degree of Node'])
    
    tree_labels,_ = ndi.label(pix_class , structure=np.ones((3, 3, 3)))#finds and labels topological islands
    
    valid_coords = np.argwhere(tree_labels > 0)
    valid_coord_labels = tree_labels[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]
    
    unique_labels = np.unique(tree_labels)
    df_average.loc[n_file,'#CC(Islands)'] = len(unique_labels)-1
    
    trees=[]
    
    for label_num, label in enumerate(unique_labels):
                #print(f'Processing label {label_num + 1} of {len(unique_labels)}')
                if label == 0:
                    continue
                global_idxs = np.argwhere(valid_coord_labels == label).flatten().tolist() #chooses one connected component
                tree = Tree(label, valid_coords[valid_coord_labels == label], global_idxs)
            
                tree.get_neighbors()
                tree.get_start_node()
                trees.append(tree)
    

    
    j= 0 
    s_deg = 0
    tips = 0
    junc = 0
    loops = 0
    
    
    for nt,tre in enumerate(trees):
        nx_graph = nx.Graph()
        for nn,neigh in enumerate(tre.neighbors):
            df_nodewise.loc[j, 'CC(Island)#'] = tre.label
            df_nodewise.loc[j, 'Node#'] = nn
            df_nodewise.loc[j, 'Degree of Node'] = len(neigh)
            if len(neigh) ==1: tips = tips +1
            if len(neigh) >=3: junc = junc +1
            s_deg = s_deg + len(neigh)
            j=j+1
            
            existing_edges = set()
            for nh in neigh:
                ed = tuple(sorted([str(nn), str(nh)]))
                if ed not in existing_edges: 
                    nx_graph.add_edge(str(nn), str(nh))
                    existing_edges.add(ed)
            
        cyc_b = list(nx.cycle_basis(nx_graph))
        print(cyc_b)
        loops = loops + len(cyc_b)
            
    df_nodewise.to_csv(os.path.join(bin_folder,file+'_nodewise.csv'), encoding='utf-8')
    
    df_average.loc[n_file,'#Nodes'] = j   
    df_average.loc[n_file,'#Tips(Deg1)'] = tips    
    df_average.loc[n_file,'#Junctions(Deg3+)'] = junc    
    df_average.loc[n_file,'Avg Deg'] = s_deg/j
    df_average.loc[n_file,'#Loops'] = loops

df_average.to_csv(os.path.join(bin_folder,'Collected_prop.csv'), encoding='utf-8')
#%%
class Tree:
    def __init__(self, label: int, voxel_idxs: np.ndarray, global_idxs):
        self.label = label
        self.voxel_idxs = voxel_idxs
        self.global_idxs = global_idxs
        self.neighbors = []
        self.start_node = None
        self.jump_distances = None
        self.nodelists = None
        self.multiscale_edge_list = None

    def get_neighbors(self):
        ckdtree = cKDTree(self.voxel_idxs)
        self.tree = ckdtree.tree
        self.neighbors = ckdtree.query_ball_point(self.voxel_idxs, r=1.733)  # a little over sqrt(3)
        self.neighbors = [np.array(neighbor) for neighbor in self.neighbors] #list of neighbor of each node + node itself: node 0 nNeigh - [0,1]
        self.neighbors = [neighbor[neighbor != i] for i, neighbor in enumerate(self.neighbors)] #removes node itself: node 0 nNeigh - [1]

    def get_start_node(self):
        # pick the first node with only one neighbor. If none exists, pick the first node
        for i, neighbor in enumerate(self.neighbors):
            if len(neighbor) == 1:
                self.start_node = i
                return
        self.start_node = 0

    # def calculate_jump_distances(self):
    #     self.jump_distances = np.full(len(self.voxel_idxs), np.inf)
    #     self.jump_distances[self.start_node] = 0

    #     stack = [(self.start_node, 0)]
    #     while stack:
    #         node, current_distance = stack.pop()
    #         for neighbor in self.neighbors[node]:
    #             if self.jump_distances[neighbor] == np.inf:  # Unvisited
    #                 self.jump_distances[neighbor] = current_distance + 1
    #                 stack.append((neighbor, current_distance + 1))

    # def generate_scale_nodelists(self, max_scale):
    #     self.nodelists = [list(range(len(self.jump_distances)))]  # All nodes for scale 0
    #     for scale in range(2, max_scale + 1):
    #         skip = 2 ** (scale - 1)
    #         valid_nodes = [i for i, dist in enumerate(self.jump_distances) if dist % skip == 0]
    #         self.nodelists.append(valid_nodes)

    # def generate_direct_accessibility(self):
    #     self.multiscale_edge_list = set()

    #     for scale_num, scale_nodelist in enumerate(self.nodelists):
    #         scale_nodelist_set = set(scale_nodelist)

    #         for node in scale_nodelist:
    #             visited = set()
    #             queue = [(node, 0)]  # (current_node, distance)

    #             while queue:
    #                 current_node, dist = queue.pop(0)
    #                 if dist > (scale_num**2 + 1):
    #                     continue
    #                 if current_node != node and current_node in scale_nodelist_set:
    #                     self.multiscale_edge_list.add((self.global_idxs[node], self.global_idxs[current_node]))
    #                     # break  # Stop after reaching the first node in the nodelist

    #                 for neighbor in self.neighbors[current_node]:
    #                     if neighbor not in visited:
    #                         visited.add(neighbor)
    #                         queue.append((neighbor, dist + 1))

