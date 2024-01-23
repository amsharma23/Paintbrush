# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:28:58 2023

@author: junlab
"""
#%% Libraries
from pycromanager import Core
from pycromanager import Acquisition
from pymmcore_plus import CMMCorePlus as pymm_Core
import time
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from pipython import GCSDevice
import os
import tifffile
#%% uManager Core loading
py_core  = pymm_Core.instance()
core = Core()
print(core)
config_file = r'C:\Program Files\Micro-Manager-2.0\Scope+SpectraX+Laser+Cam+Groups.cfg'
#core.loadSystemConfiguration(config_file)
#core.initializeAllDevices()

save_dir = r'E:\Aman\20231113'
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

dev_props = core.get_device_property_names('TIPFSOffset') #Laser box properties
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('TIPFSOffset',dev_props.get(i))))
core.set_property('TIPFSOffset','Position','180.00')
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
cam_props = core.get_device_property_names('ORCA-Fire')
for i in range(cam_props.size()):
    print(cam_props.get(i)+':'+str(core.get_property('ORCA-Fire',cam_props.get(i))))
    
#print(core.get_property('ORCA-Fire','FrameRate'))
str_arr = (core.get_allowed_property_values('ORCA-Fire','SENSOR MODE'))
for i in range(2):
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


#No 7
dev_props = core.get_device_property_names('CSUW1-Port') # State - 0/1/2,Label - Camera 1 Back,Camera 2 Side, Splitter 
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Port',dev_props.get(i))))

#No 8
dev_props = core.get_device_property_names('CSUW1-Dichroic Mirror') # State - 0/1/2,Label - Quad, Dichroic-2, Dichroic-3 
for i in range(dev_props.size()):
    print(dev_props.get(i)+':'+str(core.get_property('CSUW1-Dichroic Mirror',dev_props.get(i))))
    
#%% Laser spot check
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - PowerSetpoint (%)','0.5')
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','On')

time.sleep(10)
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')

#%% Setting device properties

def set_device_prop(g_l,cy_l,las_l, exp_t,x_p,y_p,z_p,top_left_x,top_left_y,x_size,y_size,fps):
        
    #camera
    core.set_property('ORCA-Fire','SENSOR MODE','AREA')#Global shutter cause we want to caputre high speed motion    
    core.set_roi('ORCA-Fire',top_left_x,top_left_y,x_size,y_size)
    core.set_exposure(exp_t)
    
    #Confocal
    core.set_property('CSUW1-Shutter','State','Open')
    core.set_property('CSUW1-Drive Speed','State','4000')
    core.set_property('CSUW1-Drive Speed','Run','Off')
      
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
    core.wait_for_device('ORCA-Fire')
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
        core.wait_for_device('ORCA-Fire')
        imgs.append(core.get_tagged_image())
        #print(core.get_position())
        core.set_relative_position(0.15)
        core.sleep(700)
        core.wait_for_device('TIZDrive')

    
   
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')
    core.wait_for_device('ORCA-Fire')    
    
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
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Disk','State','2')
    core.sleep(500)
    
    return(imgs)


def acq_seq_single_shot(im_nm_thr,r_im_nm,exp_t,las_on_time,z_p):
    
    imgs = []
    
    #Phase Image
    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    core.wait_for_device('ORCA-Fire')
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
        core.wait_for_device('ORCA-Fire')
        imgs.append(core.get_tagged_image())
        #print(core.get_position())
        core.set_relative_position(0.10)
        core.sleep(700)
        core.wait_for_device('TIZDrive')

    
   
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')
    core.wait_for_device('ORCA-Fire')    
    
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
                core.wait_for_device('ORCA-Fire')
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
                    core.wait_for_device('ORCA-Fire')
                    imgs.append(core.get_tagged_image())
                    #print(core.get_position())
                    core.set_relative_position(0.15)
                    core.sleep(500)
                    core.wait_for_device('TIZDrive')

                
               
                core.set_property('LightEngine','CYAN','0')
                core.set_property('LightEngine','State','0')
                core.wait_for_device('LightEngine')
                core.wait_for_device('ORCA-Fire')    
                
                core.set_property('TIPFSStatus','State','On')
                core.sleep(500)
                
                
                core.wait_for_device('TIZDrive')
                
                
                #mCh images
                core.set_property('LightEngine','GREEN','1')    
                core.set_property('LightEngine','State','1')    

                core.wait_for_device('LightEngine')
                
                core.snap_image()
                core.wait_for_device('ORCA-Fire')
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
                core.wait_for_device('ORCA-Fire')
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
                    core.wait_for_device('ORCA-Fire')
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
                        core.wait_for_device('ORCA-Fire')
                        imgs.append(core.get_tagged_image())
                        #print(core.get_position())
                        core.set_relative_position(0.15)
                        core.sleep(500)
                        core.wait_for_device('TIZDrive')

                   
                  
                    core.set_property('LightEngine','CYAN','0')
                    core.set_property('LightEngine','State','0')
                    core.wait_for_device('LightEngine')
                    core.wait_for_device('ORCA-Fire')    
                   
                    core.set_property('TIPFSStatus','State','On')
                    core.sleep(500)
                    core.wait_for_device('TIZDrive')
                   
                    #image acq after flash
                    core.set_property('LightEngine','GREEN','1')    
                    core.set_property('LightEngine','State','1')    
                    core.snap_image()
                    core.wait_for_device('ORCA-Fire')
                    imgs.append(core.get_tagged_image())
                   
                    core.set_property('LightEngine','GREEN','0')    
                    core.set_property('LightEngine','State','0')
                    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation ON
                    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to confocal
                    core.sleep(500)
                   
                    images[tm][pos] = imgs
                   
            
    return(images)                    
    
def piezo_zstack(im_nm_z,w_t,s_v,z_p):
    
    imgs = []
    
    #Phase Image
    core.set_property('NI100x DigitalIO','State','1')#Turn LED ON
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    core.wait_for_device('ORCA-Fire')
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
    
    #up ramp
    for nm_i in range(im_nm_z):
        
        core.snap_image()
        core.wait_for_device('ORCA-Fire')
        imgs.append(core.get_tagged_image())
        #print(core.get_position())
        core.set_property('AnalogIO','Volts', str(nm_i*s_v))#step of 200nm
    
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')    
    core.sleep(w_t*1000) #miliseconds     
    core.set_property('LightEngine','CYAN','1')
    core.set_property('LightEngine','State','1')  
    core.wait_for_device('LightEngine')      
    
    #down ramp    
    for nm_i in range(im_nm_z):
        
        core.snap_image()
        core.wait_for_device('ORCA-Fire')
        imgs.append(core.get_tagged_image())
        #print(core.get_position())
        core.set_property('AnalogIO','Volts', str((im_nm_z-nm_i-1)*s_v))#step of 200nm
        
    
    core.set_property('AnalogIO','Volts', '0.00')
    core.set_property('LightEngine','CYAN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')
    core.wait_for_device('ORCA-Fire')    
    
    core.set_position(z_p)
    core.sleep(500)
    core.wait_for_device('TIZDrive')
    
    core.set_property('LightEngine','GREEN','0')    
    core.set_property('LightEngine','State','0')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation ON
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')#Switch CSUW1 port to confocal
    core.set_property('CSUW1-Disk','State','2')
    core.sleep(500)
    
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
        
def image_disp_pz(image_arr,r,w_t,step_v,nm_z_steps):
    for i in range(len(image_arr)):
        im = image_arr[i]
        image = im.pix.reshape((im.tags['Height'], im.tags['Width']))

        save_d = os.path.join(save_dir,'Run_'+str(r))
        save_d = os.path.join(save_d,'wait_time_'+str(w_t)+'_Z_steps_'+str(nm_z_steps)+'_stepV_'+str(step_v))
      
        if(not(os.path.exists(save_d))):
            os.makedirs(save_d)

        file_nm =  os.path.join(save_d,str(i)+'.tif')
        tifffile.imwrite(file_nm,image,metadata=im.tags)

#%%turn laser off
core.set_property('Coherent-Scientific Remote','Laser 405-100FP - State','Off')
core.set_property('Coherent-Scientific Remote','Laser 488-100FP - State','Off')
core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-80 - State','Off')
core.set_property('LightEngine','State','0')
core.set_property('LightEngine','CYAN','0')
core.set_property('LightEngine','GREEN','0') 
core.clear_circular_buffer()
core.stop_sequence_acquisition()
#%% Main
camera_exposure = 100.0


gfP_light_level = '10.0'
mch_light_level = '15.0'
laser_pow_level = '4.0'

x_pos = core.get_x_position()
y_pos = core.get_y_position()
z_pos = core.get_position()

#x_pos_arr = [4530.70,4567.50,4570.60,4607.30,4608.30]
#y_pos_arr = [-1662.40,-1664.40,-1682.40,-1634.50,-1600.40]


#laser centre ~ [2545,985]
roi_top_lf_x = 2395
roi_top_lf_y = 835
roi_x_sz = 300
roi_y_sz = 300
fps = 200 #limit set by exposure time as fps ~ 5ms


set_device_prop(mch_light_level,gfP_light_level , laser_pow_level, camera_exposure , x_pos, y_pos,z_pos, roi_top_lf_x,roi_top_lf_y,roi_x_sz,roi_y_sz,fps)



gfp_img_interval = 100.0
num_pre_las_imgs = 3
mch_img_interval = 20
num_post_las_imgs = 100

laser_exposure_time = 200

num_imgs = 150

num_t_steps = 5
time_int_in_min = 5.0
time.sleep(1)
#images = acq_seq_cont_exposure(num_pre_las_imgs,num_post_las_imgs,camera_exposure,z_pos)
#acq_seq(g_im_nm,im_nm_thr,r_im_nm,exp_t,z_p)

#images = acq_seq_single_shot(num_pre_las_imgs,num_post_las_imgs,camera_exposure,laser_exposure_time,z_pos)
#acq_seq_single_shot(im_nm_thr,r_im_nm,exp_t,las_on_time)


#images = long_term_single_shot([x_pos], [y_pos], z_pos,laser_exposure_time,time_int_in_min,num_t_steps)
#long_term_single_shot(x_pos_arr,y_pos_arr,z_p,las_on_time,time_int_in_min,num_of_t_steps)

z_steps = 120
wait_time = 300
step_v = 0.005
images = piezo_zstack(z_steps, wait_time,step_v, z_pos)
#piezo_zstack(im_nm_z, exp_t, z_p)

#%% Image storage
run = '_3'
#for i in images:
#    print(i.tags.keys()) `

#image_disp(images,run,laser_pow_level,laser_exposure_time,num_t_steps,len([x_pos]))
image_disp_pz(images,run,wait_time,step_v,z_steps)
print('Done')