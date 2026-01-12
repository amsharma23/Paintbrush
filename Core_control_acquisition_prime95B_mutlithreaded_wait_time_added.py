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

# Threading imports
import threading
import queue


#%% uManager Core loading
core = Core()
print(core)
config_file = r'C:\Program Files\Micro-Manager-2.0\Scope+SpectraX+Laser+Cam+Groups.cfg'

save_dir = r'E:\Aman\20260111_Piezo_3752_100x_richGlu_Prime95B'
os.makedirs(save_dir,exist_ok=True)
    
#once we have confirmed we have control over the microscope and laser we will now write a custom acquisition script

#%% Check devices
#now we try and setup the imaging sequence as: test and set - Exposure time, Objective, stage position, focus position, SpectraX shutter and Coherent shutter
#handy reference - https://valelab4.ucsf.edu/~MM/doc-2.0.0-gamma/mmcorej/mmcorej/CMMCore.html
#first get list of loaded devices 
devices = core.get_loaded_devices()
for i in range(devices.size()):
    print(devices.get(i))
#%% Log file creation function
def create_acquisition_log(save_d, acq_params, device_params):
    """
    Creates a detailed log file with acquisition and device parameters.
    
    Parameters:
    -----------
    save_d : str
        Save directory path
    acq_params : dict
        Dictionary containing acquisition parameters
    device_params : dict
        Dictionary containing device properties
    """
    log_file = os.path.join(save_d, 'acquisition_log.txt')
    
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ACQUISITION LOG\n")
        f.write("=" * 80 + "\n\n")
        
        # Timestamp
        f.write(f"Acquisition Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Acquisition Parameters
        f.write("-" * 80 + "\n")
        f.write("ACQUISITION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        for key, value in acq_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Device Properties
        f.write("-" * 80 + "\n")
        f.write("DEVICE PROPERTIES\n")
        f.write("-" * 80 + "\n")
        
        for device_name, properties in device_params.items():
            f.write(f"\n[{device_name}]\n")
            for prop_name, prop_value in properties.items():
                f.write(f"  {prop_name}: {prop_value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF LOG\n")
        f.write("=" * 80 + "\n")
    
    print(f"Log file created: {log_file}")
    return log_file


def get_device_properties(core):
    """
    Retrieves current properties from all relevant devices.
    
    Parameters:
    -----------
    core : pycromanager Core
        Microscope core object
        
    Returns:
    --------
    dict : Dictionary of device properties
    """
    device_props = {}
    
    # Define devices to log
    devices_to_log = [
        'Prime95B',
        'LightEngine',
        'Coherent-Scientific Remote',
        'CSUW1-Hub',
        'CSUW1-Filter Wheel-1',
        'CSUW1-Shutter',
        'CSUW1-Drive Speed',
        'CSUW1-Bright Field',
        'CSUW1-Disk',
        'CSUW1-Port',
        'CSUW1-Dichroic Mirror',
        'TIPFSStatus',
        'AnalogIO',
        'E-709',
        'TIZDrive',
        'NI100x DigitalIO',
        'TICondenserCassette'
    ]
    
    for device in devices_to_log:
        try:
            dev_prop_names = core.get_device_property_names(device)
            device_props[device] = {}
            
            for i in range(dev_prop_names.size()):
                prop_name = dev_prop_names.get(i)
                try:
                    prop_value = core.get_property(device, prop_name)
                    device_props[device][prop_name] = prop_value
                except:
                    device_props[device][prop_name] = "N/A"
        except:
            device_props[device] = {"Error": "Device not available"}
    
    return device_props

#%% Setting default device properties

def set_device_prop(g_l,cy_l,las_l, exp_t,x_p,y_p,z_p,top_left_x,top_left_y,x_size,y_size):
        
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
    

#%% Image saving thread
def image_saver_worker(image_queue, save_d, nm_z_steps, stop_event):
    """
    Worker function that runs in a separate thread to save images.
    
    Parameters:
    -----------
    image_queue : queue.Queue
        Queue containing tuples of (image_index, tagged_image)
    save_d : str
        Save directory path
    nm_z_steps : int
        Number of z-steps per timepoint
    stop_event : threading.Event
        Event to signal when to stop the worker
    """
    print("Image saver thread started")
    saved_count = 0
    
    while not stop_event.is_set() or not image_queue.empty():
        try:
            # Get image from queue with timeout
            item = image_queue.get(timeout=0.5)
            
            if item is None:  # Poison pill to stop thread
                break
                
            ni, im = item
            
            # Save the image
            image = im.pix.reshape((im.tags['Height'], im.tags['Width']))
            
            if ni == 0:
                file_nm = os.path.join(save_d, f'channel_0_time_point_{ni}.tif')
            else:
                ni_img = ni - 1
                ni_nmz = ni_img % nm_z_steps
                t_add = math.floor(ni_img / nm_z_steps)
                tp_p = t_add
                if t_add % 2 == 1:
                    ni_nmz = nm_z_steps - 1 - ni_nmz
                
                file_nm = os.path.join(save_d, f'channel_1_time_point_{tp_p}_{ni_nmz}.tif')
            
            tifffile.imwrite(file_nm, image, metadata=im.tags)
            saved_count += 1
            
            if saved_count % 96 == 0:
                print(f"Saved {saved_count} images...")
            
            image_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error saving image: {e}")
            continue
    
    print(f"Image saver thread finished. Total images saved: {saved_count}")



    
#%% Acquisition procedures

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
    core.set_property('CSUW1-Filter Wheel-1','State','4')
    core.set_property('CSUW1-Disk','State','2')
    core.set_property('CSUW1-Drive Speed','Run','Off')#Turn disk rotation OFF
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    print('DONE Imaging')
    return(imags)

def piezo_zstack_ts_NIDAQ_threaded(tps, half_range,s_v, z_p, num_z_im, save_d, exp_t=100, wait_time_ms=0, keep_images_in_memory=False): 
    """
    Modified version that saves images in a separate thread during acquisition.
    Includes wait time between Z-stacks.
    
    Parameters:
    -----------
    tps : int
        Number of timepoints
    s_v : float
        Step voltage
    z_p : float
        Z position
    num_z_im : int
        Number of z images per timepoint
    save_d : str
        Save directory
    exp_t : float
        Exposure time in ms
    wait_time_ms : float
        Wait time between Z-stacks in milliseconds (default: 0)
    keep_images_in_memory : bool
        Whether to keep images in RAM (default: False)
    """
    
    # Create queue and thread for saving
    frame_size_mb = (roi_x_sz * roi_y_sz * 2) / 1e6  # uint16
    available_ram_mb = 8000  # leave headroom
    max_queue = int(available_ram_mb / frame_size_mb)
    image_queue = queue.Queue(maxsize=min(max_queue, 2000))
    stop_event = threading.Event()
    
    saver_thread = threading.Thread(
        target=image_saver_worker,
        args=(image_queue, save_d, num_z_im, stop_event),
        daemon=True
    )
    saver_thread.start()
    
    # Only keep images in memory if explicitly requested
    imags = [] if keep_images_in_memory else None
    bf_image = None  # Store BF separately for return
    
    # BF Image
    core.set_property('NI100x DigitalIO','State','1')
    core.wait_for_device('NI100x DigitalIO')
    
    core.snap_image()
    bf_image = core.get_tagged_image()
    if keep_images_in_memory:
        imags.append(bf_image)
    image_queue.put((0, bf_image))  # Add to save queue
    
    core.set_property('NI100x DigitalIO','State','0')
    core.wait_for_device('NI100x DigitalIO')
    
    core.set_property('TIPFSStatus','State','Off')
    core.sleep(100)
    
    # mCh images setup - build full voltage array (alternating up/down)
    max_V = (half_range * 2) * 0.1
    volt_arr_up = np.linspace(0, max_V, num_z_im)      # Up Z-stack
    volt_arr_down = volt_arr_up[::-1]                   # Down Z-stack
    
    # mCh images setup - single up-down cycle that DAQ will repeat
    max_V = (half_range * 2) * 0.1
    volt_arr_up = np.linspace(0, max_V, num_z_im)      # Up Z-stack
    volt_arr_down = volt_arr_up[::-1]                   # Down Z-stack
    
    # Single up-down cycle - DAQ will cycle through this repeatedly
    single_cycle = np.concatenate((volt_arr_up, volt_arr_down))
    
    images_per_zstack = num_z_im
    total_zstacks = tps
    total_images = images_per_zstack * total_zstacks
    
    print(f'Images per Z-stack: {images_per_zstack}')
    print(f'Total Z-stacks: {total_zstacks}')
    print(f'Total images to acquire: {total_images}')
    print(f'Wait time between Z-stacks: {wait_time_ms} ms')
    print(f'DAQ buffer size: {len(single_cycle)} samples (will cycle)')
    
    # Set up microscope for mCh imaging
    core.set_position((z_p - half_range))
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Confocal')
    core.set_property('CSUW1-Filter Wheel-1','State','4')
    core.set_property('CSUW1-Disk','State','0')
    core.set_property('CSUW1-Drive Speed','Run','On')
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
    core.sleep(500)
    
    img_count = 0
    timeout_threshold = 10.0  # seconds without progress before warning
    strt_time = time.time()
    
    # Turn on illumination once at the start
    core.set_property('LightEngine','GREEN','1')
    core.set_property('LightEngine','State','1')  
    core.wait_for_device('LightEngine')
        
    # Acquire Z-stacks with wait time between each
    try:
        # Create single NIDAQ task with one up-down cycle - will repeat automatically
        V_op_task = nidaqmx.Task()
        V_op_task.ao_channels.add_ao_voltage_chan('/Dev1/ao0')
        V_op_task.timing.cfg_samp_clk_timing(
            rate=1000.0,
            source='/Dev1/PFI0',
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=len(single_cycle)
        )       
        
        V_op_task_writer = AnalogSingleChannelWriter(V_op_task.out_stream, auto_start=False)
        V_op_task_writer.write_many_sample(single_cycle)
        V_op_task.start()
        print(f"NIDAQ task started with {len(single_cycle)} samples (CONTINUOUS mode - will cycle)")

        
        for zstack_idx in range(total_zstacks):
            core.set_property('LightEngine','GREEN','1')
            core.sleep(50)  # Ensure light is on before starting
            zstack_start_time = time.time()
            direction = "up" if zstack_idx % 2 == 0 else "down"
            
            # Start camera sequence for this Z-stack
            core.start_sequence_acquisition(images_per_zstack, 0, True)    
            last_progress_time = time.time()
            
            zstack_img_count = 0
            while(core.get_remaining_image_count() > 0 or core.is_sequence_running()):
                if(core.get_remaining_image_count() > 0):
                    img = core.pop_next_tagged_image()
                    if keep_images_in_memory:
                        imags.append(img)
                    img_count += 1
                    zstack_img_count += 1
                    last_progress_time = time.time()
                    # Add to save queue
                    image_queue.put((img_count, img))
                else:
                    # Check for timeout
                    current_time = time.time()
                    if current_time - last_progress_time > timeout_threshold:
                        print(f"WARNING: No images received for {timeout_threshold}s. Checking status...")
                        print(f"  Sequence running: {core.is_sequence_running()}")
                        print(f"  Buffer count: {core.get_remaining_image_count()}")
                        print(f"  Images acquired so far: {img_count}/{total_images}")
                        # If stuck for too long, break out
                        if current_time - last_progress_time > timeout_threshold * 3:
                            print(f"ERROR: Acquisition appears stuck. Breaking out after {img_count} images.")
                            break
                    core.sleep(exp_t / 2.0)
            
            core.stop_sequence_acquisition()
            
            zstack_elapsed = time.time() - zstack_start_time
            
            # Print progress every 10 Z-stacks or every Z-stack if wait time is long
            if zstack_idx % 10 == 0 or wait_time_ms > 1000:
                print(f"Z-stack {zstack_idx + 1}/{total_zstacks} ({direction}) complete ({zstack_img_count} images, {zstack_elapsed:.2f}s)")
                print(f"  Total images: {img_count}/{total_images}, Queue size: {image_queue.qsize()}")
            
            # Wait between Z-stacks (but not after the last one)
            if wait_time_ms > 0 and zstack_idx < total_zstacks - 1:
                core.set_property('LightEngine','GREEN','0')
                core.sleep(wait_time_ms)
        
        V_op_task.stop()
        V_op_task.close()
        print("NIDAQ task ended")
        
        end_time = time.time()
        print(f'Time for all z-stacks: {end_time - strt_time:.2f}s')
        print(f"Total images acquired: {img_count}")

    except Exception as e:
        print(f"ERROR during acquisition: {e}")
        import traceback
        traceback.print_exc()
        try:
            core.stop_sequence_acquisition()
        except:
            pass
        try:
            V_op_task.stop()
            V_op_task.close()
        except:
            pass

    # Reset microscope
    core.set_property('LightEngine','GREEN','0')
    core.set_property('LightEngine','State','0')
    core.wait_for_device('LightEngine')

    core.set_position(z_p)
    core.sleep(300)
    core.wait_for_device('TIZDrive')
    
    core.set_property('TIPFSStatus','State','On')
    core.sleep(100)
    
    core.set_property('CSUW1-Bright Field','BrightFieldPort','Bright Field')
    core.set_property('CSUW1-Filter Wheel-1','State','4')
    core.set_property('CSUW1-Disk','State','2')
    core.set_property('CSUW1-Drive Speed','Run','Off')
    core.wait_for_device('CSUW1-Drive Speed')
    core.wait_for_device('CSUW1-Bright Field')
    
    print('DONE Imaging')
    
    # Wait for all images to be saved
    print("Waiting for all images to be saved...")
    image_queue.join()  # Wait for queue to be empty
    stop_event.set()  # Signal thread to stop
    saver_thread.join(timeout=30)  # Wait for thread to finish
    
    if saver_thread.is_alive():
        print("Warning: Saver thread still running after timeout")
    else:
        print("All images saved successfully")
    
    return imags


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
camera_exposure = 175.0
print(datetime.now())

gfP_light_level = '0'
mch_light_level = '6'
laser_pow_level = '0.8'

x_pos = core.get_x_position()
y_pos = core.get_y_position()
z_pos = core.get_position()


#laser centre ~ [2326,1230]
roi_x_sz = 132                  
roi_y_sz = 100      
roi_top_lf_x = 494         
roi_top_lf_y = 598


set_device_prop(mch_light_level,gfP_light_level , laser_pow_level, camera_exposure , x_pos, y_pos,z_pos, roi_top_lf_x,roi_top_lf_y,roi_x_sz,roi_y_sz)


num_time_points = 240
z_steps = 20
wait_time =  0#in miliseconds
half_range = 3 #in um
step_v = (half_range*2)/z_steps # corresonds to 0.005~5nm; 7um

CLEAR()
images = []

#%Saving images
run = '3'
save_d = os.path.join(save_dir,'Run_'+str(run))
save_d = os.path.join(save_d, 
        f'wait_time_{wait_time:.2f}_exposure_time{camera_exposure:.2f}_half-range{half_range:.1f}_Z_steps_{z_steps:.2f}_stepV_{step_v:.2f}_number_of_TPs_{num_time_points:.2f}')
if not os.path.exists(save_d):
        os.makedirs(save_d)
print(f"Save directory: {save_d}")

# Prepare acquisition parameters
acq_params = {
    'Camera Exposure (ms)': camera_exposure,
    'Number of Time Points': num_time_points,
    'Z Steps per Time Point': z_steps,
    'Wait Time (ms)': wait_time,
    'Step Voltage (V)': step_v,
    'Step Size (µm)': step_v,  # Conversion from voltage to microns
    'Total Z Range (µm)': z_steps * step_v,
    'GFP Light Level': gfP_light_level,
    'mCherry Light Level': mch_light_level,
    'Laser Power Level (%)': laser_pow_level,
    'ROI X Size': roi_x_sz,
    'ROI Y Size': roi_y_sz,
    'ROI Top Left X': roi_top_lf_x,
    'ROI Top Left Y': roi_top_lf_y,
    'Stage X Position': x_pos,
    'Stage Y Position': y_pos,
    'Z Position': z_pos,
    'Run Number': run
}

# Get device properties
device_params = get_device_properties(core)

# Create log file
create_acquisition_log(save_d, acq_params, device_params)

st = time.time()  
images = piezo_zstack_ts_NIDAQ_threaded(num_time_points, half_range, step_v, z_pos, z_steps, save_d, camera_exposure, wait_time)
en = time.time()

with open(os.path.join(save_d, 'acquisition_log.txt'), 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("TIMING INFORMATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Acquisition Time: {en-st:.2f} seconds ({(en-st)/60:.2f} minutes)\n")
        f.write(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Average Time per Image: {(en-st)/(num_time_points*z_steps):.4f} seconds\n")
        f.write(f"Average Time per z-stack: {(en-st)/num_time_points:.2f} seconds\n")
        f.write(f"Total Images Acquired: {(images)}\n")


print('Whole time: '+str(en-st))


# %%
