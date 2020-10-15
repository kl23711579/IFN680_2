'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for more details).

# References

- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow_datasets as tfds 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# num_classes = 10
epochs = 20

class Dataset:
    '''
    This class will facilitate the creation of a few-shot dataset
    from the Omniglot dataset that can be sampled from quickly while also
    allowing to create new labels at the same time.
    
    Adopted from https://keras.io/examples/vision/reptile/
    '''
    
    def __init__(self, training):
        # Download the tfrecord files containing the omniglot data and convert to a dataset
        split = "train" if training else "test"
        ds = tfds.load('omniglot', split=split, as_supervised=True)
        # Iterate over the dataset to get each individual image and its class,
        # and put that data into a dictionary.
        self.data = {}
        
        def extraction(image, label):
            # This function will shrink the Omniglot images to the desired size,
            # scale pixel values and convert the RGB image to grayscale
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [105, 105])
            return image, label
        
        for image, label in ds.map(extraction):
            image = image.numpy()
            label = str(label.numpy())
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
            self.labels = list(self.data.keys())


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


def create_pairs(x, num_classes):
    pairs = []
    labels = []
    n = min([len(x.data[i]) for i in num_classes]) - 1
    for index, value in enumerate(num_classes):
        for i in range(n):
            z1, z2 = x.data[value][i], x.data[value][i+1]
            pairs += [[z1, z2]]
            inc = random.randrange(1, len(num_classes))
            index_n = (index + inc) % len(num_classes)
            index_v = random.randint(0, n)
            z3 = x.data[num_classes[index_n]][index_v]
            pairs += [[z1, z3]]
            labels += [1., 0.]
    return np.array(pairs), np.array(labels)


# def create_base_network(input_shape):
#     '''Base network to be shared (eq. to feature extraction).
#     '''
#     input = Input(shape=input_shape)
#     x = Flatten()(input)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(128, activation='relu')(x)
#     return Model(input, x)

def create_base_network(input_shape, num_classes):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(input, x)

# def create_base_network(input_shape, num_classes):
#     cnn_model = keras.models.Sequential()
    
#     # add layer
#     cnn_model = keras.models.Sequential()

#     # Adds layers to the sequential model
#     cnn_model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
#                      activation='relu',
#                      input_shape=input_shape))
#     cnn_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     cnn_model.add(keras.layers.Dropout(0.25))
#     cnn_model.add(keras.layers.Flatten())
#     cnn_model.add(keras.layers.Dense(128, activation='relu'))
#     cnn_model.add(keras.layers.Dropout(0.5))
#     cnn_model.add(keras.layers.Dense(num_classes, activation='softmax'))
#     return cnn_model


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# input_shape = x_train.shape[1:]

train = Dataset(training=True)
test = Dataset(training=False)

input_shape = (105, 105, 1)
labels = []
num_classes = len(train.labels) + len(test.labels)
# num_classes = len(train.labels)

# create training+test positive and negative pairs
# digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
# tr_pairs, tr_y = create_pairs(x_train, digit_indices)

# digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
# te_pairs, te_y = create_pairs(x_test, digit_indices)

tr_pairs, tr_y = create_pairs(train, train.labels)
te_pairs, te_y = create_pairs(test, test.labels)

# network definition
base_network = create_base_network(input_shape, num_classes)

base_network.summary()

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs)

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
