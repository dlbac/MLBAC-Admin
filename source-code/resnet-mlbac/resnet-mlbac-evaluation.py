
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

# Load model and data.
trainedModelName = str(sys.argv[1])
dataFileName = str(sys.argv[2])

debug = True
cols = 22

# load the dataset
raw_dataset = loadtxt(dataFileName, delimiter=' ', dtype=np.str)
dataset = raw_dataset[:,2:cols] # TO SKIP UID RID

np.random.shuffle(dataset)

# split into user-resource pair and operations variables
feature = dataset.shape[1]
if debug:
  print('Features:', feature)
metadata = feature - 4

urp = dataset[:,0:metadata]
operations = dataset[:,metadata:feature]
operations = operations.astype('float16')

#80% of total dataset is used for the evaluation
eval_size = (int)(urp.shape[0] * 0.8)

# encoding
urp = to_categorical(urp)
print('shape of URP after encoding', urp.shape)

# test data, we evaluate on these samples
urp_test = urp[:eval_size,0:feature]
operations_test = operations[:eval_size,0:4]

# training data, though this portion of samples have no use
urp_train = urp[eval_size:,0:feature]
operations_train = operations[eval_size:,0:4]

#determine batch size
batch_size = min(urp_train.shape[0]/10, 16)

x_train = urp_train[..., np.newaxis]
x_test = urp_test[..., np.newaxis]

y_train = operations_train
y_test = operations_test

PATH_MODEL = trainedModelName

if os.path.exists(PATH_MODEL):
    print('Loading trained model from {}.'.format(PATH_MODEL))
    model = load_model(PATH_MODEL)
else:
    print('No trained model found at {}.'.format(PATH_MODEL))
    exit(0)

scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


