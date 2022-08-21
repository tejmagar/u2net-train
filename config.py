import pathlib

default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)
checkpoint_path = pathlib.Path('saved/')

# Dataset
root_data_dir = pathlib.Path('data/')
dataset_dir = root_data_dir.joinpath('DUTS-TR')
image_dir = dataset_dir.joinpath('DUTS-TR-Image')
mask_dir = dataset_dir.joinpath('DUTS-TR-Mask')
