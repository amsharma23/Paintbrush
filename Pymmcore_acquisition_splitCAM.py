#%% Libraries -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 19:51:06 2025

@author: junlab
"""
from pymmcore_plus import CMMCorePlus,DeviceType
import tifffile
import nidaqmx
import time
import asyncio
from threading import Thread, Event
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional#, Callable
import logging
from pathlib import Path
#%% Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SynchronizedCameraSystem:
    def __init__(self, config_files: Optional[list[str]] = None):
        """
        Initialize synchronized camera system using separate pymmcore_plus instances
        
        Args:
            config_file: Micro-Manager configuration file path (optional)
        """
        # Initialize separate Micro-Manager cores for each camera
        self.cores: Dict[str, CMMCorePlus] = {}
        self.config_files = config_files
        
        # DAQ setup for triggering using counter output
        self.daq_device = "Dev1"  # Adjust based on your DAQ device
        self.counter_channel = "ctr0"  # Counter channel for pulse generation
        self.counter_line = f"{self.daq_device}/{self.counter_channel}"
        self.counter_output = f"{self.daq_device}/PFI12"  # Physical output pin (device specific)
        
        # Camera management
        self.cameras: List[str] = []
        self.master_camera: Optional[str] = None
        self.image_buffers: Dict[str, deque] = {}
        self.acquisition_active = Event()
        
        # Timing parameters
        self.frame_rate = 30  # Hz
        self.trigger_interval = 1.0 / self.frame_rate
        
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
        core.setProperty('LightEngine','GREEN_Intensity',20)
        core.setProperty('LightEngine','CYAN_Intensity',20)

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
                self.image_buffers[camera[0]] = deque(maxlen=100)
                
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
                        core.setProperty(camera, "TRIGGER SOURCE", "EXTERNAL")
                        core.setProperty(camera, "Trigger", "NORMAL")
                        core.setProperty(camera, "Exposure", 150.0)  # 10ms
                                                      
                    elif(camera=='Prime95B'):
                        core.setProperty(camera,"Trigger-Expose Out-Mux","1")
                        core.setProperty(camera,'ExposeOutMode','First Row')
                        core.setProperty(camera,'ShutterMode','Pre-Exposure')
                        core.setProperty(camera,"TriggerMode", "Edge Trigger")  
                        core.setProperty(camera,"Exposure", 150.0)  # 10ms           
            
                except Exception as e:
                    logger.warning(f"Could not fully configure {camera}: {e}")
                    
        except Exception as e:
            logger.error(f"Error setting up cameras: {e}")
    
    # def _setup_callbacks(self):
    #     """Setup event callbacks for image acquisition for each core"""
    #     # Connect to image ready signal for each camera core
    #     logger.info(f"Items are: {self.cores.items()}")
    #     for camera, core in self.cores.items():
    #         # Connect MDA frame ready callback
    #         core.mda.events.frameReady.connect(
    #             lambda image, metadata, cam=camera: self._on_image_ready(image, metadata, cam)
    #         )
    #         # Connect snap callback
    #         core.events.imageSnapped.connect(
    #             lambda cam=camera: self._on_image_snapped(cam)
    #         )
        
    # def _on_image_ready(self, image, metadata, camera_name):
    #     """Callback when image is ready from MDA"""
    #     self._store_image(camera_name, image, metadata)
    
    # def _on_image_snapped(self, camera_name):
    #     """Callback when image is snapped"""
    #     try:
    #         # Get the dedicated core for this camera
    #         core = self.cores[camera_name]
            
    #         # Get the latest image from this specific core
    #         image = core.getImage()
    #         metadata = {
    #             'camera': camera_name,
    #             'timestamp': datetime.now(),
    #             'frame_number': core.getImageCounter()
    #         }
    #         self._store_image(camera_name, image, metadata)
            
    #     except Exception as e:
    #         logger.error(f"Error getting snapped image from {camera_name}: {e}")
            
    # def add_image_callback(self, callback: Callable):
    #     """Add a callback function to be called when new images arrive"""
    #     self.image_callbacks.append(callback)
    
    def save_images(self, images: Dict[str, dict], save_folder: str, filename_prefix: str = None) -> Dict[str, str]:
        
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
                    if 'metadata' in image_data:
                        f.write(f"Additional metadata: {image_data['metadata']}\n")
                
            except Exception as e:
                logger.error(f"Error saving image from {camera_name}: {e}")
                
        logger.info(f"Successfully saved {len(saved_files)} images")
        return saved_files
    
    
    
    def wait_for_images(self, expected_images: int, timeout_seconds: float = 30.0) -> bool:
        """Wait for expected number of images to be captured"""
        start_time = time.time()
        
        logger.info(f"Waiting for {expected_images} images (timeout: {timeout_seconds}s)")
        
        while time.time() - start_time < timeout_seconds:
            
            status = self.get_buffer_status()
            # Check if all cameras have enough images
            all_ready = True
            for camera in self.cameras:
                if camera in status and 'remaining_images' in status[camera]:
                    if status[camera]['remaining_images'] < expected_images:
                        all_ready = False
                        break
                else:
                    all_ready = False
                    break
            
            if all_ready:
                logger.info(f"All cameras have captured {expected_images} images")
                return True
                
            # Log progress every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                progress = {cam: status[cam].get('remaining_images', 0) for cam in self.cameras}
                logger.info(f"Progress: {progress}")
                
            time.sleep(0.5)  # Check every 500ms
        
        logger.warning(f"Timeout waiting for {expected_images} images")
        final_status = self.get_buffer_status()
        logger.info(f"Final status: {final_status}")
        return False
    
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
    
    
    
    def setup_trigger_timing(self, frame_rate: float = 30):
        """Configure trigger timing parameters"""
        self.frame_rate = frame_rate
        self.trigger_interval = 1.0 / frame_rate
        logger.info(f"Trigger rate set to {frame_rate} Hz ({self.trigger_interval*1000:.1f}ms interval)")
    
    def generate_trigger_pulse(self, pulse_width_us: float = 100.0):
        """Generate a single trigger pulse via counter output"""
        logger.info("Generating single pulse")
        try:
            with nidaqmx.Task() as task:
                # Create counter output channel for pulse generation
                task.co_channels.add_co_pulse_chan_time(
                    self.counter_line,
                    name_to_assign_to_channel="",
                    units=nidaqmx.constants.TimeUnits.SECONDS,
                    idle_state=nidaqmx.constants.Level.LOW,
                    initial_delay=0.0,
                    low_time=pulse_width_us*3*(10**-6),  # Low duration
                    high_time=pulse_width_us*(10**-6)  # High duration (pulse width)
                )
                
                # Configure for single pulse
                task.timing.cfg_implicit_timing(
                    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=1  # Generate exactly 1 pulse
                )
                
                # Start and wait for completion
                task.start()
                task.wait_until_done(timeout=1.0)
                task.stop()
                
        except Exception as e:
            logger.error(f"Error generating trigger pulse: {e}")
    
    def continuous_trigger_thread(self):
        """Thread function for continuous triggering using counter output"""
        logger.info(f"Starting continuous triggering at {self.frame_rate} Hz")
        
        try:
            with nidaqmx.Task() as task:
                # Calculate timing parameters
                period_us = (10**6) / self.frame_rate  # Period in microseconds
                pulse_width_us = min(100.0, period_us * 0.1)  # 10% duty cycle, max 100µs
                
                # Create counter output for continuous pulse train
                task.co_channels.add_co_pulse_chan_freq(
                    self.counter_line,
                    name_to_assign_to_channel="Pulse Line",
                    units=nidaqmx.constants.FrequencyUnits.HZ,
                    idle_state=nidaqmx.constants.Level.LOW,
                    initial_delay=0.0,
                    freq=self.frame_rate,
                    duty_cycle=pulse_width_us / period_us  # Duty cycle as fraction
                )
                
                # Configure for continuous generation
                task.timing.cfg_implicit_timing(
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
                )
                
                logger.info(f"Counter configured: {self.frame_rate} Hz, {pulse_width_us:.1f}µs pulse width")
                
                # Start continuous generation
                task.start()
                
                # Keep running until acquisition is stopped
                while self.acquisition_active.is_set():
                    time.sleep(0.1)  # Check every 100ms
                    
        except Exception as e:
            logger.error(f"Error in trigger thread: {e}")
        finally:
            logger.info("Trigger thread stopped")
    
    def start_synchronized_acquisition(self):
        """Start synchronized acquisition across all cameras"""
        if not self.cameras:
            logger.error("No cameras found!")
            return False
            
        logger.info(f"Starting synchronized acquisition with {len(self.cameras)} cameras")
        
        try:
            # Set acquisition flag
            self.acquisition_active.set()
            
            # Start sequence acquisition for each camera using its dedicated core
            for camera in self.cameras:
                core = self.cores[camera]
                core.startContinuousSequenceAcquisition(0)  # 0 = no interval limit
                logger.info(f"Started acquisition for {camera}")
            
            # Start trigger thread
            self.trigger_thread = Thread(target=self.continuous_trigger_thread, daemon=True)
            self.trigger_thread.start()
            
            logger.info("Synchronized acquisition started!")
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

#%% Example usage and demo functions
# def image_callback_example(image_data):
#     """Example callback function for processing images"""
#     camera = image_data['camera']
#     timestamp = image_data['timestamp']
#     image_shape = image_data['image'].shape
    
#     print(f"New image from {camera}: {image_shape} at {timestamp.strftime('%H:%M:%S.%f')[:-3]}")

async def async_acquisition_demo(cam_system):
    """Demo of async acquisition monitoring"""
    print("Starting async monitoring...")
    
    for i in range(50):  # Monitor for 5 seconds
        await asyncio.sleep(0.1)
        
        images = cam_system.get_latest_images()
        sync_stats = cam_system.get_synchronization_stats()
        
        if images and 'max_sync_error_ms' in sync_stats:
            print(f"Frame {i}: {len(images)} cameras, "
                  f"sync error: {sync_stats['max_sync_error_ms']:.2f}ms "
                  f"({sync_stats['sync_quality']})")
            
def igenerate_trigger_pulse(pulse_width_us: float = 100.0):
    """Generate a single trigger pulse via counter output"""
    # DAQ setup for triggering using counter output
    idaq_device = "Dev1"  # Adjust based on your DAQ device
    icounter_channel = "ctr0"  # Counter channel for pulse generation
    icounter_line = f"{idaq_device}/{icounter_channel}"
    icounter_output = f"{idaq_device}/PFI12"  # Physical output pin (device specific)
    
    try:
        with nidaqmx.Task() as task:
            # Create counter output channel for pulse generation
            task.co_channels.add_co_pulse_chan_time(
                icounter_line,
                name_to_assign_to_channel="Pulse line",
                units=nidaqmx.constants.TimeUnits.SECONDS,
                idle_state=nidaqmx.constants.Level.LOW,
                initial_delay=0.0,
                low_time=pulse_width_us*3*(10**-6),  # Low duration
                high_time=pulse_width_us*(10**-6)  # High duration (pulse width)
            )
            
            # Configure for single pulse
            task.timing.cfg_implicit_timing(
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=1  # Generate exactly 1 pulse
            )
            
            # Start and wait for completion
            task.start()
            task.wait_until_done(timeout=1.0)
            task.stop()
            
    except Exception as e:
        logger.error(f"Error generating trigger pulse: {e}")
#%% Main
if __name__ == "__main__":
    
    config_file_P = r'C:\Micro-Manager Configuration Files\EclipseTi+Prime95B+VTRAN+Coherent_box+CSUW1_Aman.cfg'
    config_file_O = r'C:\Micro-Manager Configuration Files\ORCA_Fire.cfg'
    save_folder = r'E:\Aman\20250617_SJSC71_splitCAM_test\run4'

    #igenerate_trigger_pulse(5)
    # Example usage
    with SynchronizedCameraSystem([config_file_P,config_file_O]) as cam_system:
        
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
            if cam_system.cameras:
                master_camera = cam_system.master_camera
                
                core = cam_system.cores['Prime95B']
                
                core.setProperty('LightEngine','GREEN','1')
                core.setProperty('LightEngine','CYAN','1')
                
                core.setProperty('LightEngine','State','1')
                single_images = cam_system.single_triggered_capture()
                core.setProperty('LightEngine','State','0')
                
                core.setProperty('LightEngine','GREEN','0')
                core.setProperty('LightEngine','CYAN','0')

                print(f"\nSingle capture: {len(single_images)} images")
                
                if single_images:
                    saved_files = cam_system.save_images(single_images, save_folder)
                    print(f"Images saved to: {saved_files}")
                else:
                    print("No images captured to save")
            
            # # Option 2: Continuous synchronized acquisition
            # if cam_system.start_synchronized_acquisition():
            #     print("\nRunning continuous acquisition...")
                
            #     # Run async monitoring
            #     asyncio.run(async_acquisition_demo(cam_system))
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        #System will automatically stop acquisition when exiting context
#%%        
# core.setProperty('LightEngine','GREEN','0')
# core.setProperty('LightEngine','CYAN','0')