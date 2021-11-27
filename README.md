# optflow-cuda
This is a python script to extract TVL1 optical flow images from videos and save them in HDF5 files.
GPU acceleration and multi-processing are supported.

## Requirements
- numpy
- pillow
- cv2 (CUDA)
- h5py
- tqdm

## Installation

Singularity definition files are provided in `singularity/`.
You can build a container for this script as follows:
```bash
# CUDA 11.1
sudo singularity build env.sif singularity/env111.def

# CUDA 10.1
sudo singularity build env.sif singularity/env111.def
```

## Usage
If you use a singularity container, you can simply run `calc_flow.py` as follows:
```bash
singularity exec --nv env.sif python calc_flow.py -i <input directory> -o <output directory> -e <target extension>
```

Please see `calc_flow.py` to check other arguments.