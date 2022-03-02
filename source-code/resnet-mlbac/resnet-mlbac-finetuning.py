
from __future__ import print_function
import os
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model, to_categorical

import numpy as np
from numpy import loadtxt
from sklearn import metrics
from keras.models import Model, load_model

import os, sys


# parameters
num_classes = 4
epochs = 10
n = 1
debug = True

# Compute depth from model parameter n: depth=6n+2
depth = n * 6 + 2

########### Import File Names ######################
trainedModelName = str(sys.argv[1])
aatsFileName = str(sys.argv[2])
replayDataFileName = str(sys.argv[3])

cols = 22 # <2 uid rid> <8 user-metadata> <8 res-metadata><4 ops>


# load the AATs data
raw_dataset = loadtxt(aatsFileName, delimiter=' ', dtype=np.str)
aat_dataset = raw_dataset[:,2:cols] # TO SKIP UID RID

# load the ReplayData
replay_raw_dataset = loadtxt(replayDataFileName, delimiter=' ', dtype=np.str)
replay_dataset = replay_raw_dataset[:,2:cols] # TO SKIP UID RID

# combine them to make a common dataset
dataset = np.concatenate((aat_dataset, replay_dataset), axis=0, out=None)

np.random.shuffle(dataset)

feature = dataset.shape[1]
if debug:
  print('Features:', feature)
metadata = feature - 4

urp = dataset[:,0:metadata]
if debug:
  print('Shape of URP', urp.shape)

operations = dataset[:,metadata:feature]
operations = operations.astype('float16')

#split into training and test data
eval_size = (int)(urp.shape[0] * 0.20) #20% of total dataset

print('evaluation data size: ' + str(eval_size))

# encoding
urp = to_categorical(urp)
if debug:
  print('shape of URP after encoding', urp.shape)

# test data
urp_test = urp[:eval_size,0:feature]
operations_test = operations[:eval_size,0:4]
if debug:
  print('urp_test shape:', urp_test.shape)

# training data
urp_train = urp[eval_size:,0:feature]
operations_train = operations[eval_size:,0:4]
if debug:
  print('urp_train shape:', urp_train.shape)

#determine batch size
batch_size = min(urp_train.shape[0]/10, 16)
if debug:
  print('batch size: ' + str(batch_size))

x_train = urp_train[..., np.newaxis]
x_test = urp_test[..., np.newaxis]
if debug:
  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)

y_train = operations_train
y_test = operations_test
if debug:
  print('y_train shape:', y_train.shape)
  print('y_test shape:', y_test.shape)


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 30:
        lr *= 0.5e-3
    elif epoch > 20:
        lr *= 1e-3
    elif epoch > 6:
        lr *= 1e-2
    elif epoch > 2:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=4):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU

    full = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal')(full)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=out)

    return model

input_shape = x_train.shape[1:]

model = resnet_v1(input_shape=input_shape, depth=depth)

#Loading Weights
if os.path.exists(trainedModelName):
    if debug:
      print('Loading weights of a trained  model from {}.'.format(trainedModelName))
    model.load_weights(trainedModelName)
else:
    print('======>>>> No trained weights found! <<<<======')
    exit(0)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr_schedule(0)),
              metrics=['binary_accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks)

outputFileName = 'updated_resnet_model'
DIR_ASSETS = 'results/'
PATH_MODEL = DIR_ASSETS + outputFileName + '.hdf5'

if debug:
  print('Saving trained model to {}.'.format(PATH_MODEL))

if not os.path.isdir(DIR_ASSETS):
    os.mkdir(DIR_ASSETS)
model.save(PATH_MODEL)

result = model.predict(x_test)
result = (result > 0.5).astype(float)
test_size = y_test.shape[0]
actual_acc = 0.0
for i in range(test_size):
    actual_acc = actual_acc + np.sum(y_test[i] == result[i])

print('Accuracy Score:', actual_acc / (test_size * 4))


