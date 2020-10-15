import tensorflow as tf
import numpy as np
import random
import pickle
from tensorflow import keras
from tensorflow.keras import regularizers
from keras import backend as K

import tensorflow_datasets as tfds 

def create_cnn_model(input_shape, num_classes):
    tf.keras.backend.clear_session()
    cnn_model = keras.models.Sequential()
    
    # add layer
    cnn_model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    cnn_model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)))
    cnn_model.add(keras.layers.Dropout(0.25))
    cnn_model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    return cnn_model

# adapted from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

with np.load("/home/n10367071/IFN680_DATA/omniglot.npz") as npzfile:
    tr_x, tr_y, te_x, te_y = npzfile["tr_x"], npzfile["tr_y"], npzfile["te_x"], npzfile["te_y"]

input_shape = (28, 28, 1)

cnn_model = create_cnn_model(input_shape, 964)

input_a = tf.keras.layers.Input(shape=input_shape)
input_b = tf.keras.layers.Input(shape=input_shape)

processed_a = cnn_model(input_a)
processed_b = cnn_model(input_b)

distance = tf.keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = tf.keras.models.Model([input_a, input_b], distance)

# train
epochs=20
rms = tf.keras.optimizers.RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
history = model.fit([tr_x[:, 0], tr_x[:, 1]], tr_y,
                    batch_size=128,
                    epochs=epochs,
                    validation_data=([te_x[:, 0], te_x[:, 1]], te_y))

model.save("/home/n10367071/IFN680_A2/my_model")

#save history
with open("/home/n10367071/IFN680_A2/history.pkl", "wb") as f:
    pickle.dump(history.history, f)