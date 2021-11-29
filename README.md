# optflow-cuda
This is a python script to extract TVL1 optical flow images from videos and save them in HDF5 files.
GPU acceleration and multi-processing are supported.

## Requirements
- numpy
- pillow
- cv2 (CUDA)
- h5py
- tqdm
- ffmpeg

## Installation

Singularity definition files are provided in `singularity/`.
You can build a container for this script as follows:
```bash
# CUDA 11.1
sudo singularity build env.sif singularity/env111.def

# CUDA 10.1
sudo singularity build env.sif singularity/env101.def
```

Note that you don't have to install cuda in your computer, and only a nvidia driver is required.
If nvidia driver in your computer is old and cuda 11.1 is not supported, use `env101.def`.

## Usage
If you use a singularity container, you can simply run `calc_flow.py` as follows:
```bash
singularity exec --nv env.sif python calc_flow.py -i <input directory> -o <output directory> -e <target extension>
```

Please see `calc_flow.py` to check other arguments.

## Loading
You can load extracted optical flows and RGB frames as like:

```python
import io

import numpy
import h5py
from PIL import Image

filename = "hdf5/x.hdf5"  # path to the target file.


with h5py.File(filename, "r") as f:
    representation = "RGB"  # specify `Flow` if you want to load optical flow.
    n_frames = len(f[representation])

    # In this snippet, frame index is randomly sampled.
    frame_idx = numpy.random.randint(n_frames)
    frame_key = sorted(f[representation])[frame_idx]
    buffer = f[representation][...]

    # decode `buffer` to obtain the image.
    image_bin = io.BytesIO(buffer)
    image = Image.open(image_bin)
    image = numpy.array(image)

    # the 3rd channel of optical flow is dummy (filled by zeros).
    if representation == "Flow":
        image = image[:2]
```
