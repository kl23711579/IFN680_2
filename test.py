import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
# from sklearn.utils import shuffle


import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#For faster training during the practival, reduce the number of examples
#x_train, y_train = shuffle(x_train, y_train, random_state=0)
#x_test, y_test = shuffle(x_test, y_test, random_state=0)

#x_train = x_train[:30000]
#y_train = y_train[:30000]
#x_test = x_test[:3000]
#y_test = y_test[:3000]

img_rows, img_cols = x_train.shape[1:3]
num_classes = len(np.unique(y_train))
print(num_classes, " numclasses")
# reshape the input arrays to 4D (batch_size, rows, columns, channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
batch_size = 128
#for debugging use 3 epochs
#epochs = 3
epochs = 12

# convert to float32 and rescale between 0 and 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#
# convert class vectors to binary class matrices (aka "sparse coding" or "one hot")
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Create the model
input_layer = keras.layers.Input(input_shape)
x = keras.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape)(input_layer)
x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu', 
                                kernel_regularizer=regularizers.l2(0.01),  
                                bias_regularizer=regularizers.l1(0.01))(x)
x = keras.layers.Dropout(0.25)(x)
output = keras.layers.Dense(num_classes, activation='softmax')(x)

model = keras.models.Model(input_layer, output)
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


            
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])