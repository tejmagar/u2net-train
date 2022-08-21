import os

import random
import numpy as np

from config import *

from PIL import Image

input_files = os.listdir(image_dir)


def get_image_mask_pair(img_name, in_resize=None, out_resize=None):
    img = Image.open(image_dir.joinpath(img_name))
    mask = Image.open(mask_dir.joinpath(img_name.replace('jpg', 'png')))

    if in_resize:
        img = img.resize(in_resize[:2], Image.BICUBIC)

    if out_resize:
        mask = mask.resize(out_resize[:2], Image.BICUBIC)

    if bool(random.getrandbits(1)):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1)


def load_training_batch(batch_size=12):
    imgs = random.choices(input_files, k=batch_size)
    image_list = [get_image_mask_pair(img, in_resize=default_in_shape, out_resize=default_out_shape) for img in imgs]

    tensor_in = np.stack([i[0] / 255. for i in image_list])
    tensor_out = np.stack([i[1] / 255. for i in image_list])

    return tensor_in, tensor_out
