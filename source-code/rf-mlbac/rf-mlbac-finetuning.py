
from __future__ import print_function

import numpy as np
from numpy import loadtxt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, confusion_matrix

import os
import sys
import cloudpickle as pickle

debug = True

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

if debug:
  print('evaluation data size: ' + str(eval_size))

# encoding
scalar = StandardScaler()
scalar.fit(urp)
urp = scalar.transform(urp)

# test data
urp_test = urp[:eval_size,0:feature]
operations_test = operations[:eval_size,0:4]

# training data
urp_train = urp[eval_size:,0:feature]
operations_train = operations[eval_size:,0:4]

x_train = urp_train
x_test = urp_test
if debug:
  print('shape of train and test data')
  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)

y_train = operations_train
y_test = operations_test

if debug:
  print('y_train shape:', y_train.shape)
  print('y_test shape:', y_test.shape)

# load the trained model's weights/parameters
with open(trainedModelName + '.pkl', mode='rb') as file:
   rfmodel = pickle.load(file)

rfmodel.n_estimators += 3
rfmodel.fit(x_train, y_train)

y_preds = rfmodel.predict(x_test)
y_preds = (y_preds > 0.5).astype(int)

actual_acc = 0.0
for i in range(test_size):
    actual_acc = actual_acc + np.sum(y_test[i] == y_preds[i])

print('Accuracy Score:', actual_acc / (test_size * 4))

outputFileName = 'updated_rf_model'
DIR_ASSETS = 'results/'
PATH_MODEL = DIR_ASSETS + outputFileName + '.pkl'

if debug:
  print('Saving trained model to {}.'.format(PATH_MODEL))

if not os.path.isdir(DIR_ASSETS):
    os.mkdir(DIR_ASSETS)

model.save(PATH_MODEL)
with open(PATH_MODEL, mode='wb') as file:
   pickle.dump(rfmodel, file)


