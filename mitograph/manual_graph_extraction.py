#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:19:55 2024

@author: amansharma
"""


#%%Library load
import re
from nellie import logger
from nellie.im_info.im_info import ImInfo
from nellie.utils.general import get_reshaped_image
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import math
from tifffile import imwrite
from tifffile import imread
import os
from napari.settings import get_settings
import napari
from natsort import natsorted
import matplotlib.pyplot as plt

#%% Read all the image files - Time Series
# get_settings().application.ipy_interactive = True
# tif_files = []

# bin_folder = "/Users/amansharma/Documents/Data/Saransh_mito_data/20240521_Piezo_test_gluRich_SJSC22/Run__6-TS/wait_time_2_Z_steps_30_stepV_0.03_number_of_TPs_18/selection2"
# tif_file = os.listdir(bin_folder)
# tif_file = [file.removesuffix('.tif') for file in tif_file if file.endswith('.tif')]
# tif_files.append(tif_file[1])
# #tif_files.append(file.removesuffix('.tif') for file in tif_file if file.endswith('.tif'))

# tif_files = natsorted(tif_files)
# print(tif_files)
# #print(tif_files)

#%% Read all the image files - Population

get_settings().application.ipy_interactive = True
tif_files = []

bin_folder = "/Users/amansharma/Documents/Data/Saransh_mito_data/Mitochondria_masked_gal/"
tif_files = range(500,601)
numf = len(tif_files)
tif_files = natsorted(tif_files) #load the names of the image tiff files
print(tif_files)
#%%
def get_float_pos(st): #string parsing for extracting poistion
    st = re.split(r'[ \[\]]', st)
    pos = [int(element) for element in st if element != '']

    return pos
def load_image_and_skel(bf,idx=0,tp=0): #extract the raw 3D image and the Nellie Skeleton
    
    
    global node_path
    global nd_pdf #the nodewise dataframe
    raw_im_path = os.path.join(bf,str(tif_files[idx])+'/nellie_test/'+str(tif_files[idx])+'-ome.ome.tif')
    skel_im_path = os.path.join(bf,str(tif_files[idx])+'/nellie_test/nellie_output/'+str(tif_files[idx])+'-ome.ome-ch0-im_pixel_class.ome.tif')
    
    
    
    ts_raw_imgs = imread(raw_im_path)
    ts_skel_imgs = imread(skel_im_path)
    color_ex = []
    
    if(len(np.shape(ts_raw_imgs))!=4):
        raw_im = ts_raw_imgs
        skel_im = ts_skel_imgs 
        skel_im = np.transpose(np.nonzero(skel_im))
    
    elif(len(np.shape(ts_raw_imgs))==4):
        
        raw_im = ts_raw_imgs[tp,:,:,:]
        skel_im = ts_skel_imgs[tp,:,:,:] 
        skel_im = np.transpose(np.nonzero(skel_im))
    
    node_path = os.path.join(bf,str(tif_files[idx])+'/nellie_test/'+str(tif_files[idx])+'_etxracted.csv')
    
    
    if (os.path.exists(node_path)):
        nd_pdf = pd.read_csv(node_path)
        
        if(not(pd.isna(nd_pdf.index.max()))):
            
            
            pos_ex = nd_pdf['Position(ZXY)'].values
            print('Extraction position: '+pos_ex)
            deg_ex = nd_pdf['Degree of Node'].values.astype(int)
            pos_ex = [get_float_pos(el) for el in pos_ex]
            
            
            matching_indices = np.argwhere(np.all(skel_im[:, None, :] == pos_ex, axis=2))
            matching_indices = (matching_indices[:,0])
            print(matching_indices)
            face_color_arr = ['red' for i in range(len(skel_im))]
            
            for ni,i in enumerate(matching_indices):
              
                if deg_ex[ni] == 1: 
                    color_ex.append('blue')
                else: 
                    color_ex.append('green')


        else:
            
            pos_ex = []
            deg_ex = []
            nd_pdf =pd.DataFrame(columns=['Degree of Node','Position(ZXY)'])
            node_path = os.path.join(bf,str(tif_files[idx])+'/nellie_test/'+str(tif_files[idx])+'_etxracted.csv')
            face_color_arr = ['red' for i in range(len(skel_im))]
            
            nd_pdf.to_csv(node_path,index=False)    
        
        
    else: 
        pos_ex = []
        deg_ex = []
        nd_pdf =pd.DataFrame(columns=['Degree of Node','Position(ZXY)'])
        node_path = os.path.join(bf,str(tif_files[idx])+'/nellie_test/'+str(tif_files[idx])+'_etxracted.csv')
        face_color_arr = ['red' for i in range(len(skel_im))]

        nd_pdf.to_csv(node_path,index=False)
    
    
    
    
    
    
    
            
    return raw_im, skel_im,face_color_arr,pos_ex,color_ex

idx = 0
raw_im, skel_im,fca,imp_l,imp_c = load_image_and_skel(bin_folder,idx)
viewer = napari.view_image(raw_im, scale= [3,1,1],name=str(tif_files[idx]))
points_layer = viewer.add_points(skel_im,size=3,face_color = fca, scale= [3,1,1]) #load the skeleton in napari

if(len(imp_l)!=0):
    p_l = viewer.add_points(imp_l,size=5,face_color=imp_c,name='imp_l', scale= [3,1,1]) #load the highlighted points important layer

@viewer.bind_key('b') #Mark the branching points
def load_junction(viewer):
    ind = list(viewer.layers[1].selected_data)[0]
    
    pos =(viewer.layers[1].data[ind])
    
    insert_loc = nd_pdf.index.max()
    

    if pd.isna(insert_loc):
        insert_loc = 0    
    else:
        insert_loc = insert_loc+1
    
    nd_pdf.loc[insert_loc,'Degree of Node'] = 3
    nd_pdf.loc[insert_loc,'Position(ZXY)'] = str(pos)
    if(len(viewer.layers)>2):
        pos_ex = list(viewer.layers[2].data)
        if (any(np.array_equal(pos, arr) for arr in pos_ex)):
            pos_ex[-1] = pos
            viewer.layers[2].data = pos_ex
            color_ex = list(viewer.layers[2].face_color)
            color_ex[-1] = [0.,1.,0.,1.]
            viewer.layers[2].face_color = color_ex
            nd_pdf.to_csv(node_path,index=False)
        else:
            pos_ex.append(pos)
            viewer.layers[2].data = pos_ex
            color_ex = list(viewer.layers[2].face_color)
            color_ex[-1] = [0.,1.,0.,1.]
            viewer.layers[2].face_color = color_ex
            nd_pdf.to_csv(node_path,index=False)
    else:        
        p_l = viewer.add_points(pos,size=5,face_color=[[0.,1.,0.,1.]],name='imp_l',scale= [3,1,1])
        nd_pdf.to_csv(node_path,index=False)
    
@viewer.bind_key('t') #Mark the tip points
def load_tip(viewer):
    ind = list(viewer.layers[1].selected_data)[0]
    
    pos =(viewer.layers[1].data[ind])
    
    insert_loc = nd_pdf.index.max()
    

    if pd.isna(insert_loc):
        insert_loc = 0    
    else:
        insert_loc = insert_loc+1
    
    nd_pdf.loc[insert_loc,'Degree of Node'] = 1
    nd_pdf.loc[insert_loc,'Position(ZXY)'] = str(pos)
    if(len(viewer.layers)>2):
        pos_ex = list(viewer.layers[2].data)
        print(type(pos_ex[0]))
        if (any(np.array_equal(pos, arr) for arr in pos_ex)):
            pos_ex[-1] = pos
            viewer.layers[2].data = pos_ex
            color_ex = list(viewer.layers[2].face_color)
            color_ex[-1] = [0.,0.,1.,1.]
            viewer.layers[2].face_color = color_ex
            nd_pdf.to_csv(node_path,index=False)
            
        else:
            pos_ex.append(pos)
            viewer.layers[2].data = pos_ex
            print(list(viewer.layers[2].data))
            color_ex = list(viewer.layers[2].face_color)
            color_ex[-1] = [0.,0.,1.,1.]
            viewer.layers[2].face_color = color_ex
            nd_pdf.to_csv(node_path,index=False)

    else:        
        p_l = viewer.add_points(pos,size=5,face_color=[[0.,0.,1.,1.]],name='imp_l',scale= [3,1,1])
        nd_pdf.to_csv(node_path,index=False)

@viewer.bind_key('r') #Remove points from the important point layer
def remove_special_node(viewer):
    ind = list(viewer.layers[2].selected_data)
    nd_pdf0 = pd.read_csv(node_path)
    print(nd_pdf0)
    if(len(ind)==0):
        print('Please select points only from "imp_l"')
    else:
        pos =(viewer.layers[2].data[ind][0])
        ind = [get_float_pos(st) for st in list(nd_pdf0['Position(ZXY)'])]
        for ni,i  in enumerate(ind):
            if(all(x == y for x, y in zip(i, pos)) and len(pos) == len(i)):
                print(ni)
                nd_pdf0.drop((ni),inplace=True)
                nd_pdf0.to_csv(node_path,index=False)
                raw_im, skel_im,fca,imp_l,imp_c = load_image_and_skel(bin_folder,idx)
                viewer.layers.remove('imp_l')
                if(len(imp_l)!=0):
                    p_l = viewer.add_points(imp_l,size=5,face_color=imp_c,name='imp_l',scale= [3,1,1])


@viewer.bind_key('x') #Remove points from skeleton itself
def remove_node(viewer):
    
    ndw_pth = os.path.join(bin_folder,str(tif_files[idx])+'/nellie_test/'+str(tif_files[idx])+'-ome.ome_0_nodewise.csv')
    nd_pdf1 = pd.read_csv(ndw_pth)
    
    ind = list(viewer.layers[1].selected_data)
    pos_s = list(viewer.layers[1].data)
    pos_s = [pos_s[pos] for pos in ind]
    
    skel_pos = nd_pdf1['Position(ZXY)'].values
    skel_pos = [get_float_pos(st) for st in skel_pos]
    
    check = False
    for nsk,sk in enumerate(skel_pos):
        
        for ch in pos_s:
            if(all(x == y for x, y in zip(sk, ch)) and len(ch) == len(sk)):
                check = True
                
                isl = nd_pdf1.loc[nsk,'CC(Island)#']
                node_num = nd_pdf1.loc[nsk,'Node#']
                
                print('Island is: '+str(isl))
                indcs = nd_pdf1.index[nd_pdf1['CC(Island)#'] == isl].tolist()
                indcss = np.array(indcs)
                print(indcs)
                
                node_ind = np.where(indcss == nsk)[0][0]
                print(node_ind)
                
                neghs =  list(nd_pdf1['Neighbours'])
                neghs = [neghs[i] for i in indcss]
                neigh_0 = [re.split(r'[ \[\]\'\],]',st) for st in neghs]
                del neigh_0[node_ind] 
               
                negh = []
          
                for nn in neigh_0:
                    lisst =([element for element in nn if (element != '' and element != str(int(node_num)))])
                    print(lisst)
                    negh.append(lisst)
                
                nd_pdf1 = nd_pdf1.drop(nsk)
                del indcs[node_ind]
                
                
                for nn,n in enumerate(indcs): 
                    print(nd_pdf1.loc[n,'Neighbours'])
                    print(str(list(negh[nn])))
                    nd_pdf1.loc[n,'Neighbours'] = str(list(negh[nn]))
                
                ndw_pth_mod = os.path.join(bin_folder, str(tif_files[idx])+'/nellie_test/'+str(tif_files[idx])+'-ome.ome_0_nodewise_mod.csv')
                nd_pdf1.to_csv(ndw_pth_mod,index=False)
                
                break
        
        if check : break    
                
    
@viewer.bind_key('n') #Next image
def move_on(viewer):
    global idx
    idx = (idx+1)%numf
    viewer.layers.clear()
    raw_im, skel_im, fca,imp_l,imp_c  = load_image_and_skel(bin_folder,idx)
    
    viewer.add_image(raw_im, scale= [3,1,1],name=str(tif_files[idx]))
    points_layer = viewer.add_points(skel_im,size=3,face_color = fca,scale= [3,1,1]) #
    if(len(imp_l)!=0):
        p_l = viewer.add_points(imp_l,size=5,face_color=imp_c,name='imp_l',scale= [3,1,1])
    

@viewer.bind_key('p') # Pervious image
def move_on(viewer):
    global idx
    idx = (idx-1)%numf
    viewer.layers.clear()
    raw_im, skel_im, fca,imp_l,imp_c  = load_image_and_skel(bin_folder,idx)
    
    viewer.add_image(raw_im, scale= [3,1,1],name=str(tif_files[idx]))
    points_layer = viewer.add_points(skel_im,size=3,face_color = fca,scale= [3,1,1]) #
    if(len(imp_l)!=0):
        p_l = viewer.add_points(imp_l,size=5,face_color=imp_c,name='imp_l',scale= [3,1,1])
    
