import tifffile as tiff
import numpy as np
from ome_types import from_tiff, to_xml

def shift_tiff(input_path, output_path, zstep, dtype_out=None, dimorder_out=None):
    """
    Shifts a tiff in alternating frames
    """
    # Load the TIFF file
    with tiff.TiffFile(input_path) as tif:
        data = tif.asarray()  # Load as a NumPy array
        ome_metadata = from_tiff(input_path)
        # root = ET.fromstring(ome_metadata)
        # ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        # pixels = root.find('.//ome:Pixels', ns)
    print(np.shape(data))
    new_z_size = data.shape[1] - zstep
    shifted_data = np.zeros((data.shape[0], new_z_size, data.shape[2], 
                             data.shape[3],), dtype=data.dtype)
    for t in range(data.shape[0]):
        # shift odd frames to higher z
        if t % 2 == 1:
            #frame = data[t]
            #frame[zstep:,:,:] = frame[:-zstep,:,:]
            #frame[0:zstep,:,:] = 0
            shifted_data[t] = data[t,:-zstep,:,:]
        else:
            # Odd frames: keep upper z-slices
            shifted_data[t] = data[t, zstep:, :, :]
   
    # Update OME metadata
    ome_metadata.images[0].pixels.size_z = new_z_size
    if dtype_out:
        ome_metadata.images[0].pixels.type = dtype_out

    # Ensure dimension order is XYZT
    ome_metadata.images[0].pixels.dimension_order = 'XYCZT'
    
    updated_ome_xml = to_xml(ome_metadata)
    shifted_data = shifted_data[:, :, np.newaxis, :, :]  # Add channel dimension
    # Write as individual planes in TZCYX order for proper OME interpretation
    with tiff.TiffWriter(output_path, bigtiff=False,imagej=True) as writer:
        writer.write(
            shifted_data, 
            photometric='minisblack',
            metadata={'axes': 'TZCYX'}
        )
    print(np.shape(shifted_data))
    print(f"Shifted TIFF saved to {output_path}")


def main():
    input_tiff = f"/Users/amansharma/Desktop/20251010_Piezo_SJSC57_100x_minGal_Prime95B/Run_{2}/run{2}_all.ome.tif"
    output_tiff = f"/Users/amansharma/Desktop/20251010_Piezo_SJSC57_100x_minGal_Prime95B/Run_{2}/run{2}_all_shifted.ome.tif"

    # Specify ranges for Z, Y, and/or X dimensions (set to None to skip cropping)
    zstep = 3
    # specify the output data type
    dtype_out = "uint16"
    dimorder_out = 'XYCZT'  # Desired dimension order

    shift_tiff(input_tiff, output_tiff, zstep, dtype_out=dtype_out, dimorder_out=dimorder_out)

if __name__ == "__main__":
    main()
