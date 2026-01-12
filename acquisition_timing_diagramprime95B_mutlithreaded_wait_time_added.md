# Acquisition System Timing & Data Flow

## Hardware Signal Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HARDWARE LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐         EXPOSURE TRIGGER          ┌──────────────┐       │
│  │   Prime95B   │──────────────────────────────────►│    NI DAQ    │       │
│  │    Camera    │           (PFI0)                  │   (Dev1)     │       │
│  │              │                                   │              │       │
│  │  [sCMOS]     │                                   │  [ao0] ──────┼──┐    │
│  └──────┬───────┘                                   └──────────────┘  │    │
│         │                                                             │    │
│         │ USB 3.0                                              Analog │    │
│         │ (image data)                                        Voltage │    │
│         │                                                             │    │
│         ▼                                                             ▼    │
│  ┌──────────────┐                                   ┌──────────────┐       │
│  │      PC      │                                   │    Piezo     │       │
│  │  (pycroman)  │                                   │   (E-709)    │       │
│  └──────────────┘                                   └──────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Timing Diagram

```
TIME ──────────────────────────────────────────────────────────────────────────►

                    Z-STACK 0 (UP)              Z-STACK 1 (DOWN)           Z-STACK 2 (UP)
               │◄─────────────────────►│  │◄─────────────────────►│  │◄──────────────
               │                       │  │                       │  │
CAMERA         │                       │  │                       │  │
EXPOSURE    ┌──┐ ┌──┐ ┌──┐     ┌──┐   │  │┌──┐ ┌──┐ ┌──┐     ┌──┐│  │┌──┐ ┌──┐
(TRIG OUT)  │  │ │  │ │  │ ... │  │   │  ││  │ │  │ │  │ ... │  ││  ││  │ │  │ ...
          ──┘  └─┘  └─┘  └─────┘  └───┼──┼┘  └─┘  └─┘  └─────┘  └┼──┼┘  └─┘  └─────
               0    1    2      n-1   │W ││  0    1    2      n-1│W ││  0    1
                                      │A ││                      │A ││
                                      │I ││                      │I ││
                                      │T ││                      │T ││
                                      │  ││                      │  ││
PIEZO                                 │  ││                      │  ││
VOLTAGE     ┌─────────────────────┐   │  ││  ┌───────────────┐   │  ││  ┌─────────
(DAQ ao0)   │                  ▲  │   │  ││  │▲              │   │  ││  │
            │               ▲  │  │   │  ││  ││  ▼           │   │  ││  │
            │            ▲  │  │  │   │  ││  ││     ▼        │   │  ││  │       ▲
            │         ▲  │  │  │  │   │  ││  ││        ▼     │   │  ││  │    ▲  │
            │      ▲  │  │  │  │  │   │  ││  ││           ▼  │   │  ││  │ ▲  │  │
          ──┴───▲──┴──┴──┴──┴──┴──┴───┼──┼┴──┴┴──────────────┴───┼──┼┴──┴─┴──┴──┴──
            0V                   max_V│  │max_V              0V  │  │0V
                                      │  │                       │  │
               ◄─────────────────────►│  │◄─────────────────────►│  │
                    num_z_im frames   │  │    num_z_im frames    │  │
                    (e.g., 20)        │  │    (e.g., 20)         │  │
                                      │  │                       │  │
                                      │◄►│                       │◄►│
                                    wait_time_ms              wait_time_ms


DAQ BUFFER   [V0, V1, V2, ... Vn-1, Vn-1, Vn-2, ... V1, V0] ◄── cycles continuously
POSITION:     ▲                                          │
              └──────────────────────────────────────────┘
                        CONTINUOUS MODE (wraps around)
```

## Detailed Timing for Single Z-Stack

```
                         SINGLE Z-STACK (UP DIRECTION)
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │   Frame 0        Frame 1        Frame 2              Frame n-1  │
    │  ┌───────┐      ┌───────┐      ┌───────┐            ┌───────┐   │
    │  │exp_t  │      │exp_t  │      │exp_t  │            │exp_t  │   │
    │  │       │      │       │      │       │    ...     │       │   │
    │──┘       └──────┘       └──────┘       └────────────┘       └───│
    │                                                                  │
    │  ◄──────►                                                        │
    │  ~175ms                                                          │
    │  (exposure)                                                      │
    │                                                                  │
    │  ▼ DAQ advances on rising edge of each exposure trigger          │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
              │
              │ After all n frames acquired:
              ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                        WAIT PERIOD                               │
    │                                                                  │
    │         core.sleep(wait_time_ms)                                 │
    │                                                                  │
    │         - No camera triggers                                     │
    │         - DAQ holds at last voltage (paused, waiting for clock)  │
    │         - Images draining to disk in background                  │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
              │
              │ After wait:
              ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │              NEXT Z-STACK (DOWN DIRECTION)                       │
    │                                                                  │
    │   DAQ continues from where it left off in the buffer             │
    │   (which is the start of the "down" portion)                     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
```

## Image Data Flow (Software Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               SOFTWARE LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐                                                               │
│  │   Prime95B   │                                                               │
│  │    Camera    │                                                               │
│  └──────┬───────┘                                                               │
│         │                                                                       │
│         │ USB 3.0 (~500 MB/s theoretical)                                       │
│         │ Frame size: roi_x_sz × roi_y_sz × 2 bytes                             │
│         │ e.g., 190 × 162 × 2 = ~60 KB/frame                                    │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐           │
│  │                     Micro-Manager Core                           │           │
│  │  ┌────────────────────────────────────────────────────────────┐  │           │
│  │  │              Circular Buffer (Driver-managed)              │  │           │
│  │  │                                                            │  │           │
│  │  │   [Frame] [Frame] [Frame] [Frame] [Frame] [Frame] ...     │  │           │
│  │  │      ▲                                       │             │  │           │
│  │  │      │                                       │             │  │           │
│  │  │   write ptr                              read ptr          │  │           │
│  │  │   (camera)                            (pop_next_tagged)    │  │           │
│  │  │                                                            │  │           │
│  │  └────────────────────────────────────────────────────────────┘  │           │
│  └──────────────────────────────────────────────────────────────────┘           │
│                    │                                                            │
│                    │ core.pop_next_tagged_image()                               │
│                    │ (Main Thread)                                              │
│                    ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐           │
│  │                    Thread-Safe Queue                             │           │
│  │                   (queue.Queue)                                  │           │
│  │                                                                  │           │
│  │   maxsize = min(max_queue, 2000)                                 │           │
│  │   max_queue = available_ram_mb / frame_size_mb                   │           │
│  │                                                                  │           │
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │           │
│  │   │(i,im)│ │(i,im)│ │(i,im)│ │(i,im)│ │(i,im)│  ...              │           │
│  │   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                       │           │
│  │      ▲                                   │                       │           │
│  │   put()                               get()                      │           │
│  │   (Main)                            (Saver Thread)               │           │
│  │                                                                  │           │
│  └──────────────────────────────────────────────────────────────────┘           │
│                                         │                                       │
│                                         │ image_saver_worker()                  │
│                                         │ (Dedicated Thread)                    │
│                                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐           │
│  │                         Disk I/O                                 │           │
│  │                                                                  │           │
│  │   tifffile.imwrite(file_nm, image, metadata=im.tags)            │           │
│  │                                                                  │           │
│  │   File naming:                                                   │           │
│  │     - BF:   channel_0_time_point_0.tif                          │           │
│  │     - Fluo: channel_1_time_point_{tp}_{z_idx}.tif               │           │
│  │                                                                  │           │
│  │   z_idx is corrected for direction:                             │           │
│  │     - UP stacks:   0, 1, 2, ... n-1                             │           │
│  │     - DOWN stacks: n-1, n-2, ... 1, 0 (reversed)                │           │
│  │                                                                  │           │
│  └──────────────────────────────────────────────────────────────────┘           │
│                                         │                                       │
│                                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐           │
│  │                      Save Directory                              │           │
│  │                                                                  │           │
│  │   E:\Aman\...\Run_X\                                            │           │
│  │     ├── acquisition_log.txt                                     │           │
│  │     ├── channel_0_time_point_0.tif        (BF)                  │           │
│  │     ├── channel_1_time_point_0_0.tif      (Z-stack 0, slice 0)  │           │
│  │     ├── channel_1_time_point_0_1.tif      (Z-stack 0, slice 1)  │           │
│  │     ├── ...                                                     │           │
│  │     ├── channel_1_time_point_0_19.tif     (Z-stack 0, slice 19) │           │
│  │     ├── channel_1_time_point_1_0.tif      (Z-stack 1, slice 0)  │           │
│  │     └── ...                                                     │           │
│  │                                                                  │           │
│  └──────────────────────────────────────────────────────────────────┘           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Acquisition Loop Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MAIN ACQUISITION LOOP                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. SETUP PHASE (once)                                                         │
│      ├── Snap BF image                                                          │
│      ├── Move stage to z_p - half_range                                         │
│      ├── Configure confocal (disk on, filter wheel, etc.)                       │
│      ├── Turn on illumination                                                   │
│      ├── Start saver thread                                                     │
│      └── Start NIDAQ task (CONTINUOUS mode, single up-down cycle loaded)        │
│                                                                                 │
│   2. ACQUISITION LOOP (repeat for each Z-stack)                                 │
│      ┌─────────────────────────────────────────────────────────────────────┐    │
│      │  for zstack_idx in range(total_zstacks):                            │    │
│      │      │                                                              │    │
│      │      ├── core.start_sequence_acquisition(images_per_zstack)         │    │
│      │      │       │                                                      │    │
│      │      │       │   Camera fires exposure triggers ──► DAQ advances    │    │
│      │      │       │   Images accumulate in circular buffer               │    │
│      │      │       │                                                      │    │
│      │      │       ▼                                                      │    │
│      │      ├── while (images in buffer OR sequence running):              │    │
│      │      │       │                                                      │    │
│      │      │       ├── img = core.pop_next_tagged_image()                 │    │
│      │      │       └── image_queue.put((img_count, img))                  │    │
│      │      │                                                              │    │
│      │      ├── core.stop_sequence_acquisition()                           │    │
│      │      │                                                              │    │
│      │      └── if wait_time_ms > 0:                                       │    │
│      │              core.sleep(wait_time_ms)   ◄── WAIT BETWEEN Z-STACKS   │    │
│      │                                                                     │    │
│      └─────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│   3. CLEANUP PHASE (once)                                                       │
│      ├── Stop NIDAQ task                                                        │
│      ├── Turn off illumination                                                  │
│      ├── Return stage to original z_p                                           │
│      ├── Re-enable PFS                                                          │
│      ├── Switch back to brightfield                                             │
│      └── Wait for saver thread to finish                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Parameters

```
┌────────────────────────┬─────────────────────────────────────────────────────┐
│ Parameter              │ Description                                         │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ num_time_points (tps)  │ Total number of Z-stacks to acquire                 │
│ z_steps (num_z_im)     │ Number of Z slices per stack                        │
│ wait_time (ms)         │ Pause between each Z-stack                          │
│ half_range (µm)        │ Half of total Z range (piezo moves ±half_range)     │
│ step_v                 │ Voltage step size per Z slice                       │
│ camera_exposure (ms)   │ Exposure time per frame                             │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ images_per_zstack      │ = num_z_im                                          │
│ total_zstacks          │ = tps                                               │
│ total_images           │ = images_per_zstack × total_zstacks                 │
│ single_cycle length    │ = 2 × num_z_im (up + down voltages)                 │
└────────────────────────┴─────────────────────────────────────────────────────┘
```
HARDWARE TIMING (SYNC)                        PC SYSTEM (MEMORY & THREADS)                           STORAGE
┌───────────────────────┐             ┌──────────────────────────────────────────────────┐          ┌──────────┐
│      NIDAQ Card       │◄────────────│               Python Main Thread                 │          │          │
│ (Analog Out / Z-Piezo)│   Start     │  1. Pre-loads ALL voltages (write_many_sample)   │          │          │
│ [Volt Array Loaded]   │   Task      │  2. Triggers Camera Bursts (start_sequence...)   │          │          │
└──────────▲────────────┘             │  3. Pops images from Driver & Puts to Queue      │          │          │
           │                          └─────────────────────────┬────────────────────────┘          │          │
      PFI0 │ (TTL Clock)                                        │                                   │          │
    (Hardware Trigger)                                          │ .put()                            │          │
           │                                                    ▼                                   │          │
┌──────────┴────────────┐             ┌──────────────────────────────────────────────────┐          │          │
│   Prime95B Camera     │             │             Python Queue (RAM Buffer)            │          │          │
│                       │    USB/     │                                                  │          │          │
│  [Exposing Frame 1]   │───PCIe─────►│  [Frame 1] [Frame 2] [Frame 3] ... [Frame N]     │          │          │
│  [Exposing Frame 2]   │   Stream    │                                                  │          │          │
│          ...          │             └─────────────────────────┬────────────────────────┘          │          │
└───────────────────────┘                                       │                                   │          │
                                                                │ .get()                            │          │
                                                                ▼                                   │          │
                                      ┌──────────────────────────────────────────────────┐          │   NVMe   │
                                      │           Saver Thread (image_saver_worker)      │          │   SSD    │
                                      │                                                  │   Write  │    or    │
                                      │  1. Dequeues Image                               │─────────►│   HDD    │
                                      │  2. Reshapes Array                               │   .tif   │          │
                                      │  3. tifffile.imwrite() (BLOCKING I/O)            │          │          │
                                      └──────────────────────────────────────────────────┘          └──────────┘