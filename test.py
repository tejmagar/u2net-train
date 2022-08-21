import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from config import *
import cv2
# Training
from models.u2net import U2NET

batch_size = 8
epochs = 100
learning_rate = 0.001
weights_save_interval = 50
bce = keras.losses.BinaryCrossentropy()

def bce_loss(y_true, y_pred):
    y_p = tf.expand_dims(y_pred, axis=-1)
    loss0 = bce(y_true, y_p[0])
    loss1 = bce(y_true, y_p[1])
    loss2 = bce(y_true, y_p[2])
    loss3 = bce(y_true, y_p[3])
    loss4 = bce(y_true, y_p[4])
    loss5 = bce(y_true, y_p[5])
    loss6 = bce(y_true, y_p[6])
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6



adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
inputs = keras.Input(shape=default_in_shape)
net = U2NET()
out = net(inputs)
model = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
model.compile(optimizer=adam, loss=bce_loss, metrics=None)
model.load_weights('/home/tej/Downloads/model.h5')


def format_input(input_image):
    assert (input_image.size == default_in_shape[:2] or input_image.shape == default_in_shape)
    return np.expand_dims(np.array(input_image) / 255., 0)


image = Image.open('download.jpeg').convert('RGB')
input_image = image
if image.size != default_in_shape:
    input_image = image.resize(default_in_shape[:2], Image.BICUBIC)

input_tensor = format_input(input_image)
fused_mask_tensor = model(input_tensor, Image.BICUBIC)[0][0]
output_mask = np.asarray(fused_mask_tensor)

if image.size != default_in_shape:
    output_mask = cv2.resize(output_mask, dsize=image.size)

output_mask = np.tile(np.expand_dims(output_mask, axis=2), [1, 1, 3])
output_image = np.expand_dims(np.array(image) / 255., 0)[0]

cv2.imwrite("mask.png", output_mask * 255)

output_image = output_mask

# plt.imshow(output_image, interpolation='nearest')

img = cv2.imread("download.jpeg")
# result = cv2.cvtColor(img, cv2.COLOR_BGR2BGR)
mask = cv2.imread("mask.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# resultA = cv2.cvtColor(mask, cv2.COLOR_BGR2BGR)

# # final = cv2.bitwise_and(img, mask)
# # print(final.shape)
# # print(mask.shape)

transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
transparent[:, :, 0:3] = img
transparent[:, :, 3] = mask
cv2.imwrite("final.png", transparent)

# output_image = np.concatenate((output_mask, output_image), axis=1)


# plt.imshow(output_image, interpolation='nearest')
# img = image.copy()
# final = cv2.bitwise_and(img, output_image)
# cv2.imwrite("final.png", final)
# # plt.imshow(image * output_image, interpolation='nearest')

# cv2.imwrite(str('hh.png'), output_image * image)
