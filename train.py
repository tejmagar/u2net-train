import tensorflow as tf
from tensorflow import keras

from config import *
from dataloader import load_training_batch
from models.u2net import U2NET

tf.get_logger().setLevel('ERROR')

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


# Training
batch_size = 8
epochs = 502
learning_rate = 0.001
eval_interval = 100
weights_save_interval = 500

adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
inputs = keras.Input(shape=default_in_shape)
net = U2NET()
out = net(inputs)
model = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
model.compile(optimizer=adam, loss=bce_loss, metrics=None)
model.load_weights(checkpoint_path.joinpath('0.5873064398765564.h5'))

# train and show progress

for e in range(epochs):
    try:
        feed, out = load_training_batch(batch_size=batch_size)
        loss = model.train_on_batch(feed, out)

        print('[%s] Loss: %s' % (e, str(loss)))

        if e % weights_save_interval == 0:
            model.save_weights(str(checkpoint_path.joinpath(f'{str(loss)}.h5')))

    except KeyboardInterrupt:
        pass
    except ValueError:
        pass
