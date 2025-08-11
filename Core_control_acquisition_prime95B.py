# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 19:19:05 2025

@author: junlab
"""
#%% Libraries
from pycromanager import Core
from pycromanager import Acquisition
#from pymmcore_plus import CMMCorePlus as pymm_Core
import time
import math
import numpy as np
#from tqdm import tqdm 
#import matplotlib.pyplot as plt
#from pipython import GCSDevice
import os
import tifffile
from datetime import datetime
#NIDAQ Library
import nidaqmx
from nidaqmx.constants import VoltageUnits
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
    READ_ALL_AVAILABLE, TaskMode, TriggerType)
from nidaqmx.stream_readers import CounterReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter, AnalogSingleChannelWriter
#%% uManager Core loading
core = Core()
print(core)
config_file = r'C:\Program Files\Micro-Manager-2.0\Scope+SpectraX+Laser+Cam+Groups.cfg'
#core.loadSystemConfiguration(config_file)
#core.initializeAllDevices()

save_dir = r'D:\Aman\20250728_Piezo_SJSC88_OD0.52_100x_EGrich'
os.makedirs(save_dir,exist_ok=True)
    
#save_gfp = r'Acq_GFP'
#save_mCh = r'Acq_mCh'


#core.set_property('CoherentObis','State','0')


#once we have confirmed we have control over the microscope and laser we will now write a custom acquisition script

#%% Check devices
#now we try and setup the imaging sequence as: test and set - Exposure time, Objective, stage position, focus position, SpectraX shutter and Coherent shutter
#handy reference - https://valelab4.ucsf.edu/~MM/doc-2.0.0-gamma/mmcorej/mmcorej/CMMCore.html
#first get list of loaded devices 
devices = core.get_loaded_devices()
for i in range(devices.size()):
    print(devices.get(i))


#%%AnalogIO(NIDAQ) Piezo Z; NIDAQ analog - -10/10V=100um(range) 13 bits ---> resolution: 2mV = 20nm
#core.set_property('AnalogIO','Volts','0.07')
dev_props = core.get_device_property_names('AnalogIO') #Laser box properties
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('AnalogIO',dev_props.get(i))))
    

#%% Perfect foucs
dev_props = core.get_device_property_names('TIPFSStatus') #Laser box properties
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('TIPFSStatus',dev_props.get(i))))
core.set_property('TIPFSStatus','State','On')

# dev_props = core.get_device_property_names('TIPFSOffset') #Laser box properties
# for i in range(dev_props.size()):
#     print(dev_props.get(i)+':'+str(core.get_property('TIPFSOffset',dev_props.get(i))))
# core.set_property('TIPFSOffset','Position','180.00')
#%%E-709 Piezo
dev_props = core.get_device_property_names('E-709') #Laser box properties
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('E-709',dev_props.get(i))))

core.set_property('E-709','Send command','MOV(\'Z\',1.00)')
#%%CoherentObis Properties
dev_props = core.get_device_property_names('Coherent-Scientific Remote') #Laser box properties
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('Coherent-Scientific Remote',dev_props.get(i))))

# 488 - Port:1, 561 - Port:2, 405 - Port:3
#%%Scope Properties
dev_props = core.get_device_property_names('TIPFSStatus') # Hub has name and port number
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('TIPFSStatus',dev_props.get(i))))

#%%Camera properties
core.set_property('Prime95B','CircularBufferAutoSize','OFF')
core.set_property('Prime95B','CircularBufferFrameCount',30)
core.set_property('Prime95B','ExposeOutMode','First Row')
core.set_property('Prime95B','ShutterMode','Pre-Exposure')
#core.set_property('Prime95B','Gain','HIGH')
# core.set_property('Prime95B','OUTPUT TRIGGER PRE HSYNC COUNT','2')
# #core.set_property('Prime95B','MASTER PULSE TRIGGER SOURCE')
cam_props = core.get_device_property_names('Prime95B')
for i in range(cam_props.size()):
    print(cam_props.get(i)+':'+str(core.get_property('Prime95B',cam_props.get(i))))
    
#print(core.get_property('Prime95B','FrameRate'))


str_arr = (core.get_allowed_property_values('Prime95B','TriggerMode'))
#print(f'{str_arr}')
for i in range(3):
    print(str_arr.get(i))


#%%Light Engine properties

spec_props = core.get_device_property_names('LightEngine')
for i in range(spec_props.size()):
    print(spec_props.get(i)+':'+str(core.get_property('LightEngine',spec_props.get(i))))

#%%CSUW1 properties

#No 1 
dev_props = core.get_device_property_names('CSUW1-Hub') # Hub has name and port number
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Hub',dev_props.get(i))))
    
#No 2    
dev_props = core.get_device_property_names('CSUW1-Filter Wheel-1') # Wheel speed - 0 to 3, State and Label(wavelength)
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Filter Wheel-1',dev_props.get(i))))

#Wheel State and Label :
#0 -405
#1 -488
#2 -561
#3 -640
#4 -Blocked

#No 3
dev_props = core.get_device_property_names('CSUW1-Shutter') # State - Open, Closed
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Shutter',dev_props.get(i))))

#No 4
dev_props = core.get_device_property_names('CSUW1-Drive Speed') # Run - On/Off, State - 0 to 4000
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Drive Speed',dev_props.get(i))))
 
#No 5    
dev_props = core.get_device_property_names('CSUW1-Bright Field') # BrightFieldPort - Confocal/ Bright Field
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Bright Field',dev_props.get(i))))

#No 6
dev_props = core.get_device_property_names('CSUW1-Disk') # State - 0/1/2,Label - BrightField,Disk 1, Disk 2 
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Disk',dev_props.get(i))))

#%%
#No 7
dev_props = core.get_device_property_names('CSUW1-Port') # State - 0/1/2,Label - Camera 1 Back,Camera 2 Side, Splitter 
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Port',dev_props.get(i))))
#%%
#No 8
dev_props = core.get_device_property_names('CSUW1-Dichroic Mirror') # State - 0/1/2,Label - Quad, Dichroic-2, Dichroic-3 
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Dichroic Mirror',dev_props.get(i))))
    
#%% Laser spot check
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - PowerSetpoint (%)','1.0')
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','On')

time.sleep(100)
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')

#%% Setting default device properties

def set_device_prop(g_l,cy_l,las_l, exp_t,x_p,y_p,z_p,top_left_x,top_left_y,x_size,y_size,fps):
        
    #camera
    # core.set_property('Prime95B','SENSOR MODE','AREA')#Global shutter cause we want to caputre high speed motion 
    # core.set_property('Prime95B','OUTPUT TRIGGER KIND[2]','EXPOSURE')
    # core.set_property('Prime95B','OUTPUT TRIGGER POLARITY[2]','POSITIVE')
    # core.set_property('Prime95B','Trigger','NORMAL')
    core.set_roi('Prime95B',top_left_x,top_left_y,x_size,y_size)
    core.set_exposure('Prime95B',exp_t)
    
    #Confocal
    core.set_property('CSUW1-Shutter','State','Open')
    core.set_property('CSUW1-Drive Speed','State','4000')
    core.set_property('CSUW1-Drive Speed','Run','Off')
    core.set_property('CSUW1-Port','State','2')  
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#by default BF and not Confocal
    core.set_property('CSUW1-Disk','State','2')#use disk 0; to switch to BF switch the BFPort
    
    #illumination
    core.set_property('NI100x DigitalIO','State','0')
    core.set_property('LightEngine','State','0')

    core.set_property('LightEngine','GREEN','0')
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','GREEN_Intensity',g_l)
    core.set_property('LightEngine','CYAN_Intensity',cy_l)
    
    
    #laser box
    core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')
    core.set_property('Coherent-Scientific Remote','Laser 488-100FP - State','Off')
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-80 - State','Off')
    
    core.set_property('Coherent-Scientific Remote','Laser 405-100FP - PowerSetpoint (%)',las_l)
    #shutter laser remains 3 ie., 405
    
    #Condenser Cassette 
    core.set_property('TICondenserCassette','State','0')#use Phase3 - 100x has Ph3 ring
    
    #memory
    core.clear_circular_buffer()#usually does it on its own but just to be safe
    
    #stage position
    #core.set_xy_position(x_p,y_p)
    #core.set_position(z_p)

    
#%% Acquisition procedures
def acq_seq_cont_exposure(im_nm_thr,r_im_nm,exp_t,z_p):
    
    imgs = []
    
    #Phase Image
    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    core.wait_for_device('Prime95B')
    imgs.append(core.get_tagged_image())
    
    core.set_property('NI100x DigitalIO','State','0')
    core.wait_for_device('NI100x DigitalIO')
    
    
    #GFP images
    core.set_position((z_p-2.5))
    core.sleep(500)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Filter Wheel-1','State','4')
    core.set_property('CSUW1-Disk','State','0')
    core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
    core.sleep(500)
        
    core.set_property('LightEngine','CYAN','1')
    core.set_property('LightEngine','State','1')  
    core.wait_for_device('LightEngine')  
    for i in range(35):
        
        core.snap_image()
        core.wait_for_device('Prime95B')
        imgs.append(core.get_tagged_image())
        #print(core.get_position())
        core.set_relative_position(0.15)
        core.sleep(700)
        core.wait_for_device('TIZDrive')

    
   
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')
    core.wait_for_device('Prime95B')    
    
    core.set_position(z_p)
    core.sleep(500)
    core.wait_for_device('TIZDrive')
    
    
       
    #mCh images
    core.set_property('LightEngine','GREEN','1')    
    core.set_property('LightEngine','State','1')    

    core.wait_for_device('LightEngine')
    core.start_sequence_acquisition(im_nm_thr,0,True)    

    
    while(core.get_remaining_image_count()>0 or core.is_sequence_running()):
        if(core.get_remaining_image_count()>0):
                imgs.append(core.pop_next_tagged_image())
        
        else:
            core.sleep(exp_t/2.0)
    
    core.stop_sequence_acquisition()    
    core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','On')
    core.wait_for_device('Coherent-Scientific Remote')

    core.start_sequence_acquisition(r_im_nm,0,True)    

        
    while(core.get_remaining_image_count()>0 or core.is_sequence_running()):
        if(core.get_remaining_image_count()>0):
            imgs.append(core.pop_next_tagged_image())
            
        else:
            core.sleep(exp_t)

    core.stop_sequence_acquisition()
    
    core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')
    core.set_property('LightEngine','GREEN','0')    
    core.set_property('LightEngine','State','0')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation ON
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to BF
    core.set_property('CSUW1-Disk','State','2')
    core.sleep(500)
    
    return(imgs)


def acq_seq_single_shot(im_nm_thr,r_im_nm,exp_t,las_on_time,z_p):
    
    imgs = []
    
    #Phase Image
    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    core.wait_for_device('Prime95B')
    imgs.append(core.get_tagged_image())
    
    core.set_property('NI100x DigitalIO','State','0')
    core.wait_for_device('NI100x DigitalIO')
    
    #GFP images
    core.set_position((z_p-2.5))
    core.sleep(500)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Filter Wheel-1','State','4')
    core.set_property('CSUW1-Disk','State','0')
    core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
   
    
    core.sleep(500)
    
    
    core.set_property('LightEngine','CYAN','1')
    core.set_property('LightEngine','State','1')  
    core.wait_for_device('LightEngine')  
    for i in range(35):
        
        core.snap_image()
        core.wait_for_device('Prime95B')
        imgs.append(core.get_tagged_image())
        #print(core.get_position())
        core.set_relative_position(0.10)
        core.sleep(700)
        core.wait_for_device('TIZDrive')

    
   
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')
    core.wait_for_device('Prime95B')    
    
    core.set_position(z_p)
    core.sleep(500)
    core.wait_for_device('TIZDrive')
    
    
    #mCh images
    core.set_property('LightEngine','GREEN','1')    
    core.set_property('LightEngine','State','1')    

    core.wait_for_device('LightEngine')
    core.start_sequence_acquisition(im_nm_thr,0,True)    

    
    while(core.get_remaining_image_count()>0 or core.is_sequence_running()):
        if(core.get_remaining_image_count()>0):
                imgs.append(core.pop_next_tagged_image())
        
        else:
            core.sleep(exp_t)
    
    #laser flash
    core.stop_sequence_acquisition()    
    core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','On')
    core.wait_for_device('Coherent-Scientific Remote')
    core.sleep(las_on_time)
    #core.wait_for_device('CoherentObis')
    core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')
    #core.wait_for_device('CoherentObis')

    #image acq after flash
    core.start_sequence_acquisition(r_im_nm,0,True)    

        
    while(core.get_remaining_image_count()>0 or core.is_sequence_running()):
        if(core.get_remaining_image_count()>0):
            imgs.append(core.pop_next_tagged_image())
            
        else:
            core.sleep(exp_t+20)

    core.stop_sequence_acquisition()
    
    core.set_property('LightEngine','GREEN','0')    
    core.set_property('LightEngine','State','0')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation ON
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to confocal
    core.sleep(500)
    
    return(imgs)
            
def long_term_single_shot(x_pos_arr,y_pos_arr,z_p,las_on_time,time_int_in_min,num_of_t_steps):
    
    l = len(x_pos_arr)
    images = []#1st index time point, 2nd index pos index, then take GFP z_stack and one plane in mCh
    for t in range(num_of_t_steps):
        images.append([[]for i in range(l)])
    
    
    for tm in range(num_of_t_steps):
        print('t',tm)
        core.sleep(time_int_in_min*60.0*1000)
        print('Done for:'+str((tm+1)*time_int_in_min)+'min out of '+str((num_of_t_steps)*time_int_in_min)+'min'+' Time stamp:'+str(time.time()))
        for pos in range(len(x_pos_arr)):
            print('p',pos)
            
            #core.set_xy_position(x_pos_arr[pos],y_pos_arr[pos])
            
            core.sleep(500)
            if(tm == 0):
                
                imgs = []
                
                #BF Image
                core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
                core.wait_for_device('NI100x DigitalIO')
                
                core.snap_image()
                core.wait_for_device('Prime95B')
                imgs.append(core.get_tagged_image())
                
                core.set_property('NI100x DigitalIO','State','0')
                core.wait_for_device('NI100x DigitalIO')
                core.set_property('TIPFSStatus','State','Off')
                
                #GFP images
                core.set_position((z_p-3.0))
                core.sleep(500)
                
                core.wait_for_device('TIZDrive')
                
                core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
                core.set_property('CSUW1-Filter Wheel-1','State','4')
                core.set_property('CSUW1-Disk','State','0')
                core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
                core.wait_for_device('CSUW1-Drive Speed')
                core.wait_for_device('CSUW1-Bright Field')
                
               
                
                core.sleep(500)
                
                
                core.set_property('LightEngine','CYAN','1')
                core.set_property('LightEngine','State','1')  
                core.wait_for_device('LightEngine')  
                for i in range(45):
                    
                    core.snap_image()
                    core.wait_for_device('Prime95B')
                    imgs.append(core.get_tagged_image())
                    #print(core.get_position())
                    core.set_relative_position(0.15)
                    core.sleep(500)
                    core.wait_for_device('TIZDrive')

                
               
                core.set_property('LightEngine','CYAN','0')
                core.set_property('LightEngine','State','0')
                core.wait_for_device('LightEngine')
                core.wait_for_device('Prime95B')    
                
                core.set_property('TIPFSStatus','State','On')
                core.sleep(500)
                
                
                core.wait_for_device('TIZDrive')
                
                
                #mCh images
                core.set_property('LightEngine','GREEN','1')    
                core.set_property('LightEngine','State','1')    

                core.wait_for_device('LightEngine')
                
                core.snap_image()
                core.wait_for_device('Prime95B')
                imgs.append(core.get_tagged_image())
                
                core.set_property('LightEngine','State','0')   
                
                #laser flash
                core.stop_sequence_acquisition()    
                core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','On')
                core.wait_for_device('Coherent-Scientific Remote')
                core.sleep(las_on_time)
                #core.wait_for_device('CoherentObis')
                core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')
                #core.wait_for_device('CoherentObis')
                
                #image acq after flash
                core.set_property('LightEngine','State','1')
                core.snap_image()
                core.wait_for_device('Prime95B')
                imgs.append(core.get_tagged_image())
                
                core.set_property('LightEngine','GREEN','0')    
                core.set_property('LightEngine','State','0')
                core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation ON
                core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to confocal
                core.sleep(500)
               
                images[tm][pos] = imgs
            else:
                
                    imgs = []
                   
                    #BF Image
                    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
                    core.wait_for_device('NI100x DigitalIO')
                   
                    core.snap_image()
                    core.wait_for_device('Prime95B')
                    imgs.append(core.get_tagged_image())
                   
                    core.set_property('NI100x DigitalIO','State','0')
                    core.wait_for_device('NI100x DigitalIO')
                    core.set_property('TIPFSStatus','State','Off')
                    #GFP images
                    core.set_position((z_p-3.0))
                   
                    core.wait_for_device('TIZDrive')
                   
                    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
                    core.set_property('CSUW1-Filter Wheel-1','State','4')
                    core.set_property('CSUW1-Disk','State','0')
                    core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
                    core.wait_for_device('CSUW1-Drive Speed')
                    core.wait_for_device('CSUW1-Bright Field')
                   
                  
                   
                    core.sleep(500)
                   
                   
                    core.set_property('LightEngine','CYAN','1')
                    core.set_property('LightEngine','State','1')  
                    core.wait_for_device('LightEngine')  
                    for i in range(45):
                       
                        core.snap_image()
                        core.wait_for_device('Prime95B')
                        imgs.append(core.get_tagged_image())
                        #print(core.get_position())
                        core.set_relative_position(0.15)
                        core.sleep(500)
                        core.wait_for_device('TIZDrive')

                   
                  
                    core.set_property('LightEngine','CYAN','0')
                    core.set_property('LightEngine','State','0')
                    core.wait_for_device('LightEngine')
                    core.wait_for_device('Prime95B')    
                   
                    core.set_property('TIPFSStatus','State','On')
                    core.sleep(500)
                    core.wait_for_device('TIZDrive')
                   
                    #image acq after flash
                    core.set_property('LightEngine','GREEN','1')    
                    core.set_property('LightEngine','State','1')    
                    core.snap_image()
                    core.wait_for_device('Prime95B')
                    imgs.append(core.get_tagged_image())
                   
                    core.set_property('LightEngine','GREEN','0')    
                    core.set_property('LightEngine','State','0')
                    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation ON
                    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to confocal
                    core.sleep(500)
                   
                    images[tm][pos] = imgs
                   
            
    return(images)                    
    
def piezo_zstack_slower(im_nm_z,w_t,s_v,z_p): 
    
    imgs = []
    
    
    
    #BF Image
    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    core.wait_for_device('Prime95B')
    imgs.append(core.get_tagged_image())
    
    core.set_property('NI100x DigitalIO','State','0')
    core.wait_for_device('NI100x DigitalIO')
    
    core.set_property('TIPFSStatus','State','Off')
    core.sleep(100)
    
    #GFP images
    core.set_position((z_p-2.5))
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Filter Wheel-1','State','4')#Dual Channel
    core.set_property('CSUW1-Disk','State','0')
    core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
    core.sleep(500)
        
    core.set_property('LightEngine','GREEN','1')
    core.set_property('LightEngine','State','1')  
    core.wait_for_device('LightEngine')  
    
    #up ramp
    strt_time = time.time()
    # core.start_sequence_acquisition(im_nm_z,0,True)    
 
    # while(core.get_remaining_image_count()>0 or core.is_sequence_running()):
    #     if(core.get_remaining_image_count()>0):
    #         imgs.append(core.pop_next_tagged_image())
            
    #     else:
    #         core.sleep(10)
    for nm_i in range(im_nm_z):
        core.snap_image()
        imgs.append(core.get_tagged_image())
        core.wait_for_device('Prime95B')
        core.set_property('AnalogIO','Volts', str(nm_i*s_v))#step of 100nm
    
    
    core.set_property('AnalogIO','Volts', '0.00')
    core.set_property('LightEngine','GREEN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')

        
    #     #print(str(nm_i*s_v))
    end_time = time.time()
    print('Time for z-stack: '+str(end_time-strt_time))  
    # for im in range(im_nm_z):
    #     print(im)
    #     imgs.append(core.get_tagged_image())
        
    
   
    print('Number of images are: '+str(len(imgs)))
    
    
    
    
    
    
    
    
    #down ramp    
    # for nm_i in range(im_nm_z):
        
    #     core.snap_image()
    #     core.wait_for_device('Prime95B')
    #     imgs.append(core.get_tagged_image())
    #     #print(core.get_position())
    #     core.set_property('AnalogIO','Volts', str((im_nm_z-nm_i-1)*s_v))#step of 200nm
    #     #print(str((im_nm_z-nm_i-1)*s_v))
    
   
    
    core.set_position(z_p+0.02)
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to BF
    core.set_property('CSUW1-Filter Wheel-1','State','0')
    core.set_property('CSUW1-Disk','State','2')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation OFF
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    print('DONE Imaging')
    return(imgs)



def piezo_zstack_NIDAQ(s_v,z_p,exp_t=100): 
    
    imgs = []
    
    
    
    #BF Image
    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    core.wait_for_device('Prime95B')
    imgs.append(core.get_tagged_image())
    
    core.set_property('NI100x DigitalIO','State','0')#LED OFF
    core.wait_for_device('NI100x DigitalIO')
    
    core.set_property('TIPFSStatus','State','Off')
    core.sleep(100)
    
    #GFP images
    half_range = 2.5 # in um
    max_V = (half_range*2.0)*0.1 # full_range(um)*0.1 = full_range(V) 
    print(max_V/s_v)
    num_z_im = round(max_V/s_v)
    volt_arr = np.linspace(0,max_V,num_z_im)
    im_nm_z = len(volt_arr)
    print('Number of Z-slices is: '+str(im_nm_z))
    
    core.set_position((z_p-half_range))
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Filter Wheel-1','State','4')#Dual Channel
    core.set_property('CSUW1-Disk','State','0')
    core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
    core.sleep(500)
        
    core.set_property('LightEngine','CYAN','1')
    core.set_property('LightEngine','State','1')  
    core.wait_for_device('LightEngine')  
    
    #up ramp
    
    with nidaqmx.Task() as V_op_task:    
   #    V_op_task.ci_channels.add_ci_count_edges_chan('/Dev1/ctr0',name_to_assign_to_channel="Counter_FRAME", edge=Edge.RISING,initial_count=0,count_direction=CountDirection.EXTERNAL_SOURCE)
   #    V_op_task.ci_channels.all.ci_count_edges_term = '/Dev1/PFI0'
           
    
       V_op_task.ao_channels.add_ao_voltage_chan('/Dev1/ao0')
       V_op_task.timing.cfg_samp_clk_timing(rate=10.0,source='/Dev1/PFI12',sample_mode=AcquisitionType.CONTINUOUS)       
       
       V_op_task_writer = AnalogSingleChannelWriter(V_op_task.out_stream,auto_start=False)
       V_op_task_writer.write_many_sample(volt_arr)
       V_op_task.start()
   #    print(V_op_task.read())
       
       core.start_sequence_acquisition(im_nm_z-1,0,True)    
       strt_time = time.time()        
       while(core.get_remaining_image_count()>0 or core.is_sequence_running()):
           if(core.get_remaining_image_count()>0):
                   imgs.append(core.pop_next_tagged_image())
           
           else:
               core.sleep(exp_t/2.0)
       end_time = time.time()
       core.stop_sequence_acquisition()    
       

       
   #    print(V_op_task.read())
       V_op_task.stop()
       V_op_task.close()
    
    
    
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')

    print('Time for z-stack: '+str(end_time-strt_time))  
   
        

   
    
    core.set_position(z_p+0.02)
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to BF
    core.set_property('CSUW1-Filter Wheel-1','State','0')
    core.set_property('CSUW1-Disk','State','2')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation OFF
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    print('DONE Imaging')
    return(imgs)

def piezo_zstack_ts_NIDAQ(tps,s_v,z_p,num_z_im,exp_t=100): 
    
    imags = []
    
    
    
    #BF Image
    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    #core.wait_for_device('Prime95B')
    imags.append(core.get_tagged_image())
    
    core.set_property('NI100x DigitalIO','State','0')#LED OFF
    core.wait_for_device('NI100x DigitalIO')
    
    core.set_property('TIPFSStatus','State','Off')
    core.sleep(100)
    
    #mCh images
    half_range = 3.5 # in um
    max_V = (half_range*2)*0.1 # full_range(um)*0.1 = full_range(V) 
    #print(max_V/s_v)
    #num_z_im = round(max_V/s_v)
    volt_arr = np.linspace(0,max_V,num_z_im)
    volt_arr_rev = volt_arr[::-1]; #volt_arr_rev = volt_arr_rev[:-1]

    full_arr = np.concatenate((volt_arr, volt_arr_rev))
    full_arr_rep = full_arr
    for i in range(math.floor(tps/2)-1):
        full_arr_rep = np.concatenate((full_arr_rep , full_arr))
    im_nm_z = len(full_arr_rep)
    z_slices = len(full_arr)
    print('Number of Z-slices is: '+str(z_slices))
    
    print(full_arr)
    
      #set-up mCh
    core.set_position((z_p-half_range))
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Filter Wheel-1','State','4')#Dual Channel
    core.set_property('CSUW1-Disk','State','0')
    core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
    core.sleep(500)
        
      
    
      #NIDAQ commands
    with nidaqmx.Task() as V_op_task:    
   #    V_op_task.ci_channels.add_ci_count_edges_chan('/Dev1/ctr0',name_to_assign_to_channel="Counter_FRAME", edge=Edge.RISING,initial_count=0,count_direction=CountDirection.EXTERNAL_SOURCE)
   #    V_op_task.ci_channels.all.ci_count_edges_term = '/Dev1/PFI0'
           
    
       V_op_task.ao_channels.add_ao_voltage_chan('/Dev1/ao0')
       V_op_task.timing.cfg_samp_clk_timing(rate=10.0,source='/Dev1/PFI0',sample_mode=AcquisitionType.CONTINUOUS)       
       
       V_op_task_writer = AnalogSingleChannelWriter(V_op_task.out_stream,auto_start=False)
       V_op_task_writer.write_many_sample(full_arr_rep)
       V_op_task.start()
   #    print(V_op_task.read())
       core.set_property('LightEngine','GREEN','1')
       core.set_property('LightEngine','State','1')  
       core.wait_for_device('LightEngine')
       
       core.start_sequence_acquisition(im_nm_z,0,True)    
       strt_time = time.time()
       while(core.get_remaining_image_count()>0 or core.is_sequence_running()):
           if(core.get_remaining_image_count()>0):
               imags.append(core.pop_next_tagged_image())
           else:
                core.sleep(exp_t/2.0)
                
       core.stop_sequence_acquisition()
       end_time = time.time()
       print('Time for z-stacks: '+str(end_time-strt_time))
       
        #core.sleep(wait_t)

       
   #    print(V_op_task.read())
       V_op_task.stop()
       V_op_task.close()
    
    
      #reset
    core.set_property('LightEngine','GREEN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')

    core.set_position(z_p)
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('TIPFSStatus','State','On')
    core.sleep(100)
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to BF
    core.set_property('CSUW1-Filter Wheel-1','State','0')
    core.set_property('CSUW1-Disk','State','2')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation OFF
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    print('DONE Imaging')
    return(imags)

def piezo_zstack_PSF(im_nm_z,w_t,s_v,z_p):
    
    imgs = []
    
        
    
    #GFP images
    core.set_position((z_p-1.5))
    core.sleep(500)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Filter Wheel-1','State','4')
    core.set_property('CSUW1-Disk','State','0')
    core.set_property('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
    core.sleep(500)
        
    core.set_property('LightEngine','CYAN','1')
    core.set_property('LightEngine','State','1')  
    core.wait_for_device('LightEngine')  
    
    for nums in range(1):
        print(nums)
        #up ramp
        for nm_i in range(im_nm_z):
            
            core.snap_image()
            core.wait_for_device('Prime95B')
            imgs.append(core.get_tagged_image())
            #print(core.get_position())
            core.set_property('AnalogIO','Volts', str(nm_i*s_v))
        
        core.set_property('LightEngine','CYAN','0')
        core.set_property('LightEngine','State','0')
        core.wait_for_device('LightEngine')    
        core.sleep(w_t*100) #miliseconds to seconds     
        core.set_property('LightEngine','CYAN','1')
        core.set_property('LightEngine','State','1')  
        core.wait_for_device('LightEngine')      
        
        #down ramp    
        for nm_i in range(im_nm_z):
            
            core.snap_image()
            core.wait_for_device('Prime95B')

            imgs.append(core.get_tagged_image())
            #print(core.get_position())
            core.set_property('AnalogIO','Volts', str((im_nm_z-nm_i-1)*s_v))#step of 200nm
        
    
    core.set_property('AnalogIO','Volts', '0.00')
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to BF
    core.set_property('CSUW1-Filter Wheel-1','State','0')
    core.set_property('CSUW1-Disk','State','2')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation OFF
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')  
    
    core.set_position(z_p)
    core.sleep(500)
    core.wait_for_device('TIZDrive')
    
    return(imgs)


#%%Save images
def image_disp(image_arr,r,l_pow_lvl,l_exp_tm,num_of_t_steps,pos_num):
    for i in range(num_of_t_steps):
        for j in range(pos_num):
            im = image_arr[i][j]
        #image = Image.fromarray(image)
        #print(descp)
            for nm_i,imms in enumerate(im):
                image = imms.pix.reshape((imms.tags['Height'], imms.tags['Width']))
        
                save_d = os.path.join(save_dir,'Run_'+str(r))
                save_d = os.path.join(save_d,'Time_id_'+str(i)+' Pos_id_'+str(j))
                if(not(os.path.exists(save_d))):
                    os.makedirs(save_d)
        
                file_nm =  os.path.join(save_d,str(nm_i)+'_p-'+str(l_pow_lvl)+'_e-'+str(l_exp_tm)+'_zs-'+str(0.1)+'.tif')
        
            
                try:
                    print(str(nm_i)+' time of acquisiton: '+imms.tags['ElapsedTime-ms'])            
                
                    tifffile.imwrite(file_nm,image,metadata=imms.tags)
            
                except KeyError:
                    print('No elapsed time')
                    tifffile.imwrite(file_nm,image,metadata=imms.tags)


def image_disp_pz_zstack(image_arr,save_d,nm_z_steps):
    for ni,im in enumerate(image_arr):
        image = im.pix.reshape((im.tags['Height'], im.tags['Width']))
        file_nm =  os.path.join(save_d,(str(ni)+'_zstep_'+str(ni%nm_z_steps)+'.tif'))
        tifffile.imwrite(file_nm,image,metadata=im.tags)
    print('Done Saving')
        
def image_disp_pz_ts_zstack(image_arr,save_d,tp,nm_z_steps):
    
    if(tp==0):
        im = image_arr
        image = im.pix.reshape((im.tags['Height'], im.tags['Width']))
        file_nm =  os.path.join(save_d,('channel_0'+'_time_point_'+str(tp)+'.tif'))
        tifffile.imwrite(file_nm,image,metadata=im.tags)
    else:        
            ni = tp-1
            im = image_arr
            image = im.pix.reshape((im.tags['Height'], im.tags['Width']))
            ni_nmz = ni%nm_z_steps; t_add =  math.floor(ni/nm_z_steps); tp_p = t_add;
            if(t_add%2 ==1): ni_nmz = nm_z_steps-1 - ni_nmz
            #print(tp_p,ni_nmz,nm_z_steps)
            file_nm =  os.path.join(save_d,('channel_1'+'_time_point_'+str(tp_p)+'_'+str(ni_nmz)+'.tif'))
            tifffile.imwrite(file_nm,image,metadata=im.tags)
    
#%%turn laser off
def CLEAR():
    core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')
    core.set_property('Coherent-Scientific Remote','Laser 488-100FP - State','Off')
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-80 - State','Off')
    core.set_property('LightEngine','State','0')
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','GREEN','0')    
    images = []
    core.clear_circular_buffer()
    core.stop_sequence_acquisition()
    print('Reset')
#%% Reset
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')
core.set_property('Coherent-Scientific Remote','Laser 488-100FP - State','Off')
core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-80 - State','Off')
core.set_property('LightEngine','State','0')
core.set_property('LightEngine','CYAN','0')
core.set_property('LightEngine','GREEN','0')    
images = []
core.clear_circular_buffer()
core.stop_sequence_acquisition()


#%% Main
camera_exposure = 150.0
print(datetime.now())

gfP_light_level = '0.0'
mch_light_level = '17.0'
laser_pow_level = '0.8'

x_pos = core.get_x_position()
y_pos = core.get_y_position()
z_pos = core.get_position()


#laser centre ~ [2326,1230]
roi_x_sz = 156                
roi_y_sz = 133
roi_top_lf_x = 541       
roi_top_lf_y = 471
fps = 200 #limit set by exposure time as fps ~ 5ms


set_device_prop(mch_light_level,gfP_light_level , laser_pow_level, camera_exposure , x_pos, y_pos,z_pos, roi_top_lf_x,roi_top_lf_y,roi_x_sz,roi_y_sz,fps)

# gfp_img_interval = 100.0
# num_pre_las_imgs = 3
# mch_img_interval = 20
# num_post_las_imgs = 100

# laser_exposure_time = 200

# num_imgs = 150

# num_t_steps = 5
# time_int_in_min = 5.0
#time.sleep(1)
#images = acq_seq_cont_exposure(num_pre_las_imgs,num_post_las_imgs,camera_exposure,z_pos)
#acq_seq(g_im_nm,im_nm_thr,r_im_nm,exp_t,z_p)

#images = acq_seq_single_shot(num_pre_las_imgs,num_post_las_imgs,camera_exposure,laser_exposure_time,z_pos)
#acq_seq_single_shot(im_nm_thr,r_im_nm,exp_t,las_on_time)


#images = long_term_single_shot([x_pos], [y_pos], z_pos,laser_exposure_time,time_int_in_min,num_t_steps)
#long_term_single_shot(x_pos_arr,y_pos_arr,z_p,las_on_time,time_int_in_min,num_of_t_steps)

num_time_points = 150
z_steps = 30
wait_time =  0#in miliseconds
step_v = 7/z_steps # corresonds to 0.005~50nm; 6um



CLEAR()
images = []
st = time.time()  
#def piezo_zstack_ts_NIDAQ(tps,s_v,z_p,wait_t,num_z_im,exp_t=100): 
images = piezo_zstack_ts_NIDAQ(num_time_points,step_v,z_pos,z_steps,camera_exposure)    
en = time.time()
print('Whole time: '+str(en-st))
# for i in range(num_time_points):
#     print('Time_point: '+str(i))
#     st = time.time()     
    
#     image_list.append(images)
#     en = time.time()
#     print('Z-stack and saving time: '+str(en-st))
#     time.sleep(wait_time)
 
# for ni,imgs in enumerate(image_list):
#     image_disp_pz_ts_zstack(imgs,save_d,ni,len(imgs))
#     print('Done')

#%% Save NIDAQ z-stacks TS

print(len(images))
#%%Saving images
run = '10'
save_d = os.path.join(save_dir,'Run_'+str(run))
save_d = os.path.join(save_d,'wait_time_'+str(wait_time)+'_exposure_time'+str(camera_exposure)+'_Z_steps_'+str(z_steps)+'_stepV_'+str(step_v)+'_number_of_TPs_'+str(num_time_points)+'_whole_time_'+str(en-st))
if(not(os.path.exists(save_d))):
    os.makedirs(save_d)
print(save_d)
for ni,i in enumerate(images):    
    image_disp_pz_ts_zstack(i,save_d,ni,30)
print('Done Saving')    

# %%
