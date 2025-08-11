#%% Libraries -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:51:06 2025

@author: junlab
"""
from pymmcore_plus import CMMCorePlus,DeviceType
import tifffile
import numpy as np
import time
import math
from threading import Thread, Event
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional#, Callable
import logging
from pathlib import Path
import os
#Nidaq library
import nidaqmx
#from nidaqmx.constants import VoltageUnits
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge, READ_ALL_AVAILABLE, TaskMode, TriggerType)
#from nidaqmx.stream_readers import CounterReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter #, AnalogMultiChannelWriter

#%% Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SynchronizedCameraSystem:
    def __init__(self, config_files: Optional[list[str]] = None,exposure: float = 100.0, laser_int: float = 45.0,rois: list = [[[0,0],[1200,1200]],[[0,0],[4432,2368]]]):
        """
        Initialize synchronized camera system using separate pymmcore_plus instances
        
        Args:
            config_file: Micro-Manager configuration file path (optional)
        """
        
        #Setup RoIs
        self.rois = rois
        self.roi_p = self.rois[0]
        self.roi_o = self.rois[1]
        self.laser_int = laser_int

        # Initialize separate Micro-Manager cores for each camera
        self.cores: Dict[str, CMMCorePlus] = {}
        self.config_files = config_files
        
        # DAQ setup for triggering using counter output
        self.daq_device = "Dev1"  # Adjust based on your DAQ device
        self.counter_channel = "ctr0"  # Counter channel for pulse generation
        self.counter_line = f"/{self.daq_device}/{self.counter_channel}"
        self.counter_output = f"/{self.daq_device}/PFI12"  # Physical output pin (device specific)
        
        # Piezo control setup using DAQ analog output
        self.piezo_ao_channel = f"/{self.daq_device}/ao0"  # Analog output for piezo
        self.camera_trigger_source = f"/{self.daq_device}/Ctr0Out"  # Camera trigger input
        self.piezo_voltage_array = None  # Will hold the piezo voltage waveform
        self.piezo_sample_rate = 2000.0/(exposure)  # Hz - adjust as needed; slightly higher than the frame rate
        
        
        # Camera management
        self.cameras: List[str] = []
        self.master_camera: Optional[str] = None
        self.image_buffers: Dict[str, deque] = {}
        self.acquisition_active = Event()
        
        # Timing parameters
        self.exposure = exposure  # ms
        self.trigger_interval = self.exposure + 10 #ms 
        
        # Callbacks and handlers
        #self.image_callbacks: List[Callable] = [] #List of functions - callables, i.e. things that can be "called"
        
        self._setup_cameras()
        self._setup_CSUW1()
        #self._setup_callbacks()
    
        
    
    def _setup_CSUW1(self):
        
        core = self.cores['Prime95B']
        
        #Confocal
        core.setProperty('CSUW1-Shutter','State','Open')
        core.setProperty('CSUW1-Bright Field','BrightFieldPort','Confocal')#Switch CSUW1 port to confocal
        core.setProperty('CSUW1-Port','State','0')# 0 - splitter
        core.setProperty('CSUW1-Filter Wheel-1','State','4')
        core.setProperty('CSUW1-Filter Wheel-2','State','1') #2-GFP filter rest blocked
        core.setProperty('CSUW1-Disk','State','0')
        core.setProperty('CSUW1-Drive Speed','Run','On')#Turn disk rotation ON
        core.waitForDevice('CSUW1-Drive Speed')
        core.waitForDevice('CSUW1-Bright Field')
        
        #illumination
        core.setProperty('NI100x DigitalIO','State','0')
        core.setProperty('LightEngine','State','0')

        core.setProperty('LightEngine','GREEN','0')
        core.setProperty('LightEngine','CYAN','0')
        core.setProperty('LightEngine','GREEN_Intensity',laser_int)
        core.setProperty('LightEngine','CYAN_Intensity',laser_int)

    def _setup_cameras(self):
        """Discover and configure cameras with separate core instances"""
        try:
            
            # Create separate core instances for each camera
            for i, config in enumerate(self.config_files):
                
                # Create dedicated core for this camera
                camera_core = CMMCorePlus()
                if self.config_files:
                    camera_core.loadSystemConfiguration(config)
                
                # Set this camera as the active camera for this core
                camera = camera_core.getLoadedDevicesOfType(DeviceType.CameraDevice)
                camera_core.setCameraDevice(camera[0])
                
                # Store the core and camera info
                self.cores[camera[0]] = camera_core
                self.cameras.append(camera[0])
                self.image_buffers[camera[0]] = deque(maxlen=200)
                
                # Set the first camera as master
                if camera[0] == 'Prime95B':
                    self.master_camera = camera[0]
                
                logger.info(f"Found camera: {camera[0]} with dedicated core")
                
                
            # Configure cameras for external triggering
            for camera in self.cameras:
                try:
                    core = self.cores[camera]
                    # Set camera-specific properties using dedicated core
                    # Try to set external trigger mode (device-specific)
                    if(camera=='ORCA-Fire'):
                        core.setROI(self.roi_o[0][0], self.roi_o[0][1], self.roi_o[1][0], self.roi_o[1][1])
                        core.setProperty(camera, "TRIGGER SOURCE", "EXTERNAL")
                        core.setProperty(camera, "Trigger", "NORMAL")
                        core.setProperty(camera, "Exposure", self.exposure)  # 10ms
                                                      
                    elif(camera=='Prime95B'):
                        core.setROI(self.roi_p[0][0], self.roi_p[0][1], self.roi_p[1][0], self.roi_p[1][1])
                        core.setProperty(camera,"Trigger-Expose Out-Mux","1")
                        core.setProperty(camera,'ExposeOutMode','First Row')
                        core.setProperty(camera,'ShutterMode','Pre-Exposure')
                        core.setProperty(camera,"TriggerMode", "Edge Trigger")  
                        core.setProperty(camera,"Exposure", self.exposure)  # 10ms           
            
                except Exception as e:
                    logger.warning(f"Could not fully configure {camera}: {e}")
                    
        except Exception as e:
            logger.error(f"Error setting up cameras: {e}")
    

    def save_image(self, images: Dict[str, dict], save_folder: str, filename_prefix: str = None) -> Dict[str, str]:
        
        saved_files = {}
        
        # Create save folder if it doesn't exist
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename if no prefix provided
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"capture_{timestamp}"
        
        logger.info(f"Saving {len(images)} images to {save_folder}")
        
        for camera_name, image_data in images.items():
            try:
                # Extract image array
                image_array = image_data['image']
                
                # Create filename
                filename = f"{filename_prefix}_{camera_name}.tiff"
                file_path = save_path / filename
                
                # Save as TIFF file
                tifffile.imwrite(str(file_path), image_array)
                
                # Store the saved file path
                saved_files[camera_name] = str(file_path)
                
                logger.info(f"Saved {camera_name} image: {file_path} (shape: {image_array.shape})")
                
                # Optionally save metadata as text file
                metadata_filename = f"{filename_prefix}_{camera_name}_metadata.txt"
                metadata_path = save_path / metadata_filename
                
                with open(metadata_path, 'w') as f:
                    f.write(f"Camera: {camera_name}\n")
                    f.write(f"Timestamp: {image_data['timestamp']}\n")
                    f.write(f"Image shape: {image_array.shape}\n")
                    f.write(f"Image dtype: {image_array.dtype}\n")
                    f.write(f"Laser Intesity: {self.laser_int}\n")
                    if 'metadata' in image_data:
                        f.write(f"Additional metadata: {image_data['metadata']}\n")
                
            except Exception as e:
                logger.error(f"Error saving image from {camera_name}: {e}")
                
        logger.info(f"Successfully saved {len(saved_files)} images")
        return saved_files
    
    def save_images(self, images: Dict[str, list[dict]], save_folder: str, filename_prefix: str = None) -> Dict[str, str]:
        
        
        saved_files = {}
        for camera_name in images.keys():
            
            # Create save folder if it doesn't exist
            save_path = Path(os.path.join(save_folder,camera_name))
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {len(images[camera_name])} images to {save_path}")
            saved_files[camera_name] = []
            for i_n,image_data in enumerate(images[camera_name]):    
        
                try:
                    # Extract image array
                    image_array = image_data['image']
                    
                    # Create filename
                    filename = f"{i_n}_{camera_name}.tiff"
                    save_path_tif = Path(os.path.join(save_path,'Tiff'))
                    save_path_tif.mkdir(parents=True, exist_ok=True)
                    
                    file_path = save_path_tif / filename
                    
                    # Save as TIFF file
                    tifffile.imwrite(str(file_path), image_array)
                    
                    # Store the saved file path
                    saved_files[camera_name].append(str(file_path))
                    
                    #logger.info(f"Saved {camera_name} image: {file_path} (shape: {image_array.shape})")
                    
                    # Optionally save metadata as text file
                    metadata_filename = f"{i_n}_{camera_name}_metadata.txt"
                    save_path_m = Path(os.path.join(save_path,'Metadata'))
                    save_path_m.mkdir(parents=True, exist_ok=True)
                    metadata_path = save_path_m / metadata_filename
                    
                    with open(metadata_path, 'w') as f:
                        f.write(f"Camera: {camera_name}\n")
                        f.write(f"Timestamp: {image_data['timestamp']}\n")
                        f.write(f"Image shape: {image_array.shape}\n")
                        f.write(f"Image dtype: {image_array.dtype}\n")
                        f.write(f"Laser Intesity: {self.laser_int}\n")
                        if 'metadata' in image_data:
                            f.write(f"Additional metadata: {image_data['metadata']}\n")
                    
                except Exception as e:
                    logger.error(f"Error saving image from {camera_name}: {e}")
                
            logger.info(f"Successfully saved {len(saved_files[camera_name])} images")
        
        return saved_files
    
    def retrieve_images_from_camera(self, camera: str, max_images: int = None) -> List[dict]:
        """Retrieve images from camera's internal circular buffer"""
        images = []
        
        try:
            core = self.cores[camera]
            
            # Get number of images in buffer
            remaining_images = core.getRemainingImageCount()
            logger.info(f"{camera}: {remaining_images} images in buffer")
            
            if max_images is not None:
                remaining_images = min(remaining_images, max_images)
            
            # Retrieve images from buffer
            for i in range(remaining_images):
                try:
                    # Get next image from buffer
                    image = core.popNextImage()
                    
                    # Create metadata
                    metadata = {
                        'camera': camera,
                        'timestamp': datetime.now(),
                        'buffer_index': i,
                        'frame_number': len(images)
                    }
                    
                    image_data = {
                        'camera': camera,
                        'image': image,
                        'metadata': metadata,
                        'timestamp': datetime.now()
                    }
                    
                    images.append(image_data)
                    
                except Exception as e:
                    logger.warning(f"Error retrieving image {i} from {camera}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error retrieving images from {camera}: {e}")
            
        logger.info(f"Retrieved {len(images)} images from {camera}")
        return images

    def retrieve_all_images(self, max_images_per_camera: int = None) -> Dict[str, List[dict]]:
        """Retrieve images from all cameras' internal buffers"""
        all_images = {}
        # multi_images = self.retrieve_all_images()
        # for k in multi_images.keys():
        #     saved_files = self.save_images()
        #     logger.info(f"Images saved to: {saved_files}")
        # else:
        #     logger.error("No images captured to save")     
        for camera in self.cameras:
            all_images[camera] = self.retrieve_images_from_camera(camera, max_images_per_camera)
            
        return all_images
    
    def get_buffer_status(self) -> Dict[str, dict]:
        """Get buffer status for all cameras"""
        status = {}
        
        for camera in self.cameras:
            try:
                core = self.cores[camera]
                status[camera] = {
                    'remaining_images': core.getRemainingImageCount(),
                    'sequence_running': core.isSequenceRunning(),
                    'buffer_free': core.isBufferOverflowed() == False,
                    'total_memory_mb': core.getBufferTotalCapacity(),
                    'free_memory_mb': core.getBufferFreeCapacity()
                }
                
            except Exception as e:
                status[camera] = {'error': str(e)}
                
        return status
    
    
    
    def generate_trigger_pulse(self, pulse_width_us: float = 10.0):
        """Generate a single trigger pulse via counter output"""
        logger.info("Generating single pulse")
       
        
        try:
            with nidaqmx.Task() as trigger_task:
                # Create counter output channel for pulse generation
                trigger_task.co_channels.add_co_pulse_chan_time(
                    self.counter_line,
                    name_to_assign_to_channel="Triggering line",
                    units=nidaqmx.constants.TimeUnits.SECONDS,
                    idle_state=nidaqmx.constants.Level.LOW,
                    initial_delay=0.0,
                    low_time=pulse_width_us*3*(10**-6),  # Low duration
                    high_time=pulse_width_us*(10**-6)  # High duration (pulse width)
                )
                
                # Configure for single pulse
                trigger_task.timing.cfg_implicit_timing(
                    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=1  # Generate exactly 1 pulse
                )
                
                
                
                # Start and wait for completion
                trigger_task.start()
                trigger_task.wait_until_done(timeout=1.0)
                trigger_task.stop()
                
        except Exception as e:
            logger.error(f"Error generating trigger pulse: {e}")
    
    def continuous_trigger_thread(self,save_folder,full_arr_rep: list = np.arange(0.0,0.63,0.03)):
        
        """Thread function for continuous triggering using counter output"""
        logger.info(f"Starting continuous triggering at {1000/self.trigger_interval} Hz")
        logger.info(f"Number of total images per camera should be {len(full_arr_rep)}")
        #logger.info(f"{full_arr_rep}")
        try:
            with nidaqmx.Task() as trigger_task, nidaqmx.Task() as piezo_task:
                
                # Calculate timing parameters
                period_s = self.trigger_interval * 0.001 # Period in microsecond (ms to s)
                logger.info(f"Pulse period {period_s}")
                pulse_width_s = min(0.01, period_s * 0.1)  # 10% duty cycle, max 100µs
                logger.info(f"Pulse width {pulse_width_s}")
                
                # Create counter output for continuous pulse train
                trigger_task.co_channels.add_co_pulse_chan_time(
                self.counter_line,
                name_to_assign_to_channel="Pulse Line",
                units=nidaqmx.constants.TimeUnits.SECONDS,
                idle_state=nidaqmx.constants.Level.LOW,
                initial_delay=0.0,
                low_time= period_s - pulse_width_s,  # Time signal is low
                high_time = pulse_width_s              # Time signal is high (pulse width)
                )
                
                # Configure for continuous generation
                trigger_task.timing.cfg_implicit_timing(
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=len(full_arr_rep)  
                )
                
                
                piezo_task.ao_channels.add_ao_voltage_chan(self.piezo_ao_channel)
                # Use camera trigger as clock source for piezo
                piezo_task.timing.cfg_samp_clk_timing(
                    rate=self.piezo_sample_rate,
                    source=self.camera_trigger_source,  # Sync to camera trigger
                    sample_mode=AcquisitionType.CONTINUOUS
                )
                piezo_task_writer = AnalogSingleChannelWriter(piezo_task.out_stream,auto_start=False)
                piezo_task_writer.write_many_sample(full_arr_rep)
                piezo_task.start()
                
                logger.info(f"Counter configured: {1000/self.trigger_interval} Hz, {pulse_width_s:.3f}s pulse width")
                
                core = self.cores[self.master_camera]
                
                core.setProperty('LightEngine','GREEN','1')
                core.setProperty('LightEngine','CYAN','1')
                core.setProperty('LightEngine','State','1')
                # Start continuous generation
                trigger_task.start()
                
                time.sleep(self.exposure*len(full_arr_rep)/1000)
                
                while not trigger_task.is_task_done():
                    logger.info("Waiting")
                    time.sleep(self.exposure/500)
                    
                piezo_task.stop() 
                trigger_task.stop()
                self.acquisition_active.clear()
                logger.info("Triggering stopped")                
                
                core.setProperty('LightEngine','State','0')
                core.setProperty('LightEngine','GREEN','0')
                core.setProperty('LightEngine','CYAN','0')
                
                
                multi_images = self.retrieve_all_images()
                self.save_images(multi_images,save_folder)
                
            
                
                
        except Exception as e:
            logger.error(f"Error in trigger thread: {e}")
        finally:
            logger.info("Trigger thread stopped")
    
    def start_synchronized_acquisition(self,save_folder, tps: int = 1, volt_arr: list = np.arange(0.0,0.63,0.03)):
        """Start synchronized acquisition across all cameras"""
        if not self.cameras:
            logger.error("No cameras found!")
            return False
            
        logger.info(f"Starting synchronized acquisition with {len(self.cameras)} cameras")
        
        
        volt_arr_rev = volt_arr[::-1]; #volt_arr_rev = volt_arr_rev[:-1]
        full_arr = np.concatenate((volt_arr, volt_arr_rev))

        full_arr_rep = full_arr
        
        for i in range(math.floor(tps/2)-1):
            full_arr_rep = np.concatenate((full_arr_rep , full_arr))
        
        
        try:
            # Set acquisition flag
            self.acquisition_active.set()
            
            # Start sequence acquisition for each camera using its dedicated core
            for camera in self.cameras:
                core = self.cores[camera]
                core.startContinuousSequenceAcquisition(0)  # 0 = no interval limit
                logger.info(f"Started acquisition for {camera}")
            
            
            # Start trigger thread
            logger.info("Synchronized acquisition started!")
            self.trigger_thread = Thread(target=self.continuous_trigger_thread(save_folder,full_arr_rep), daemon=True)
            self.trigger_thread.start()
        
            return True
            
        except Exception as e:
            logger.error(f"Error starting acquisition: {e}")
            self.stop_acquisition()
            return False
    
    def stop_acquisition(self):
        """Stop all acquisition and triggering"""
        logger.info("Stopping acquisition...")
        
        # Clear acquisition flag
        self.acquisition_active.clear()
        
        # Stop all cameras using their dedicated cores
        for camera in self.cameras:
            try:
                core = self.cores[camera]
                core.stopSequenceAcquisition()
                logger.info(f"Stopped acquisition for {camera}")
            except Exception as e:
                logger.warning(f"Error stopping {camera}: {e}")
        
        # Wait for trigger thread to finish
        if hasattr(self, 'trigger_thread') and self.trigger_thread.is_alive():
            self.trigger_thread.join(timeout=2.0)
            if self.trigger_thread.is_alive():
                logger.warning("Trigger thread did not stop cleanly")
        
        logger.info("Acquisition stopped")
    
    def get_latest_images(self) -> Dict[str, dict]:
        """Get the most recent images from all cameras"""
        images = {}
        
        for camera in self.cameras:
            if camera in self.image_buffers and self.image_buffers[camera]:
                # Get the most recent image
                images[camera] = self.image_buffers[camera][-1]
                
        return images
    
    def get_all_buffered_images(self) -> Dict[str, List[dict]]:
        """Get all buffered images from all cameras"""
        all_images = {}
        
        for camera in self.cameras:
            all_images[camera] = list(self.image_buffers[camera])
            
        return all_images
    
    def clear_image_buffers(self):
        """Clear all image buffers"""
        for camera in self.cameras:
            self.image_buffers[camera].clear()
    
    def single_triggered_capture(self) -> Dict[str, dict]:
        """Capture a single synchronized frame from all cameras"""
        captured_images = {}
        
        # Method 1: Try external trigger with sequence acquisition
        try:
            logger.info("Attempting DAQ triggered sequence acquisition...")
            
            # Prepare all cameras for single shot
            for camera in self.cameras:
                core = self.cores[camera]
                
                # Stop any ongoing acquisition
                if core.isSequenceRunning():
                    core.stopSequenceAcquisition()
                    time.sleep(0.1)
                
                # Clear buffer
                core.clearCircularBuffer()
                
                # Start single image sequence acquisition
                core.startSequenceAcquisition(1, 0, True)  # 1 image, no interval, stop on overflow
                logger.info(f"Started single sequence for {camera}")
            
            # Give camera time to be ready
            time.sleep(0.5)
        
            # Generate trigger pulse
            self.generate_trigger_pulse(pulse_width_us=10.0)
            
            # Wait for images
            if self.wait_for_images(1, timeout_seconds=5.0):
                # Retrieve images
                for camera in self.cameras:
                    images = self.retrieve_images_from_camera(camera, max_images=1)
                    if images:
                        captured_images[camera] = images[0]
                        logger.info(f"Retrieved triggered image from {camera}")
            
            # Stop sequence acquisitions
            for camera in self.cameras:
                core = self.cores[camera]
                if core.isSequenceRunning():
                    core.stopSequenceAcquisition()
        
        except Exception as e:
            logger.error(f"Triggered capture failed: {e}")
        
        logger.info(f"Captured {len(captured_images)} synchronized images")
        return captured_images
        
        
    def get_camera_properties(self, camera: str) -> dict:
        """Get current properties of a camera"""
        properties = {}
        try:
            core = self.cores[camera]
            # Get common properties
            prop_names = core.getDevicePropertyNames(camera)
            
            for pn,prop in enumerate(prop_names):
                properties[prop] = core.getProperty(camera,prop)
                
        except Exception as e:
            logger.error(f"Error getting properties for {camera}: {e}")
            
        return properties
    
    def set_camera_property(self, camera: str, property_name: str, value):
        """Set a property for a specific camera"""
        try:
            core = self.cores[camera]
            if core.hasProperty(camera, property_name):
                core.setProperty(camera, property_name, value)
                logger.info(f"Set {camera}.{property_name} = {value}")
            else:
                logger.warning(f"Property {property_name} not found for {camera}")
                    
        except Exception as e:
            logger.error(f"Error setting {camera}.{property_name}: {e}")
    
    def get_synchronization_stats(self) -> dict:
        """Get statistics about synchronization quality"""
        latest_images = self.get_latest_images()
        
        if len(latest_images) < 2:
            return {"error": "Need at least 2 cameras for sync stats"}
        
        timestamps = [img['timestamp'] for img in latest_images.values()]
        
        # Calculate sync statistics
        time_diffs = [(t - min(timestamps)).total_seconds() * 1000 for t in timestamps]  # ms
        
        stats = {
            'num_cameras': len(latest_images),
            'max_sync_error_ms': max(time_diffs) - min(time_diffs),
            'timestamps': {cam: img['timestamp'].isoformat() 
                          for cam, img in latest_images.items()},
            'sync_quality': 'Good' if max(time_diffs) - min(time_diffs) < 5.0 else 'Poor'
        }
        
        return stats
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_acquisition()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_acquisition()
            # Clean up all cores
            for core in self.cores.values():
                del core
        except:
            pass


#%% Main
if __name__ == "__main__":
    
    config_file_P = r'C:\Micro-Manager Configuration Files\EclipseTi+Prime95B+VTRAN+Coherent_box+CSUW1_Aman.cfg'
    config_file_O = r'C:\Micro-Manager Configuration Files\ORCA_Fire.cfg'
    save_folder = r'D:\Aman\20250718_SJSC71_minGal_OD0.41\run17_zsteps22_tp40'
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    #igenerate_trigger_pulse(5)
    # Example usage
    exposure_time = 90
    laser_int = 50
    roi_P = [[627,454],[115,115]]
    roi_O = [[1876,792],[300,312]]
    
    rois = [roi_P,roi_O]
    with SynchronizedCameraSystem([config_file_P,config_file_O],exposure_time,laser_int,rois) as cam_system:
        
        # Add image processing callback
        #cam_system.add_image_callback(image_callback_example)
        
        # Set trigger rate
        #cam_system.setup_trigger_timing(frame_rate=20)  # 20 Hz
        
        try:
            # Show camera info
            # for camera in cam_system.cameras:
            #     props = cam_system.get_camera_properties(camera)
            #     print(f"\n{camera} properties: {props}")
                
            # Option 1: Single synchronized capture
            # Use the specific camera's core to snap an image
            # if cam_system.cameras:
            #     master_camera = cam_system.master_camera
                
            #     core = cam_system.cores['Prime95B']
                
            #     core.setProperty('LightEngine','GREEN','1')
            #     core.setProperty('LightEngine','CYAN','1')
                
            #     core.setProperty('LightEngine','State','1')
            #     single_images = cam_system.single_triggered_capture()
            #     core.setProperty('LightEngine','State','0')
                
            #     core.setProperty('LightEngine','GREEN','0')
            #     core.setProperty('LightEngine','CYAN','0')

            #     print(f"\nSingle capture: {len(single_images)} images")
                
            #     if single_images:
            #         saved_files = cam_system.save_images(single_images, save_folder)
            #         print(f"Images saved to: {saved_files}")
            #     else:
            #         print("No images captured to save")
            
            # Option 2: Continuous synchronized acquisition
            z_range = 6 #in um
            z_steps = 20
            step_v = (z_range*0.1)/z_steps # corresonds to 0.005~50nm; 6um
            volt_arr = np.arange(0.0,z_range*0.11,step_v)
            print("Steps are: "+str(len(volt_arr)))
            time_points = 40
            zz_steps = len(volt_arr)
            #print(full_arr_rep)
            core = cam_system.cores[cam_system.master_camera]
            z_pos = core.getPosition()
            core.setPosition(z_pos - (z_range/2))
            cam_system.start_synchronized_acquisition(save_folder,time_points,volt_arr)
            #time.sleep((exposure_time/1000)*z_steps*time_points*2)
            core.setPosition(z_pos)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        #System will automatically stop acquisition when exiting context
#%%


#%% Example usage and demo functions

            
# def igenerate_trigger_pulse(pulse_width_us: float = 100.0):
#     """Generate a single trigger pulse via counter output"""
#     # DAQ setup for triggering using counter output
#     idaq_device = "Dev1"  # Adjust based on your DAQ device
#     icounter_channel = "ctr0"  # Counter channel for pulse generation
#     icounter_line = f"{idaq_device}/{icounter_channel}"
#     icounter_output = f"{idaq_device}/PFI12"  # Physical output pin (device specific)
    
#     try:
#         with nidaqmx.Task() as task:
#             # Create counter output channel for pulse generation
#             task.co_channels.add_co_pulse_chan_time(
#                 icounter_line,
#                 name_to_assign_to_channel="Pulse line",
#                 units=nidaqmx.constants.TimeUnits.SECONDS,
#                 idle_state=nidaqmx.constants.Level.LOW,
#                 initial_delay=0.0,
#                 low_time=pulse_width_us*3*(10**-6),  # Low duration
#                 high_time=pulse_width_us*(10**-6)  # High duration (pulse width)
#             )
            
#             # Configure for single pulse
#             task.timing.cfg_implicit_timing(
#                 sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
#                 samps_per_chan=1  # Generate exactly 1 pulse
#             )
            
#             # Start and wait for completion
#             task.start()
#             task.wait_until_done(timeout=1.0)
#             task.stop()
            
#     except Exception as e:
#         logger.error(f"Error generating trigger pulse: {e}")