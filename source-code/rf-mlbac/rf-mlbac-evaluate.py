

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

trainedModelName = str(sys.argv[1])
dataFileName = str(sys.argv[2])

# load the dataset
raw_dataset = loadtxt(dataFileName, delimiter=' ', dtype=np.str)
dataset = raw_dataset[:,2:cols] # TO SKIP UID RID

np.random.shuffle(dataset)

feature = dataset.shape[1]
if debug:
  print('Features:', feature)
attribs = feature - 4

urp = dataset[:,0:attribs]
operations = dataset[:,attribs:feature]
operations = operations.astype('float16')

#80% of total dataset is used for the evaluation
eval_size = (int)(urp.shape[0] * 0.80)
if debug:
  print('evaluation data size: ' + str(eval_size))

# encoding
scalar = StandardScaler()
scalar.fit(urp)
urp = scalar.transform(urp)

# test data, we evaluate on these samples
urp_test = urp[:eval_size,0:feature]
operations_test = operations[:eval_size,0:4]

# training data, though this portion of samples have no use
urp_train = urp[eval_size:,0:feature]
operations_train = operations[eval_size:,0:4]

x_train = urp_train
x_test = urp_test
if debug:
  print('x_train shape:', x_train.shape)
  print('x_test shape:', x_test.shape)

y_train = operations_train
y_test = operations_test

if debug:
  print('y_train shape:', y_train.shape)
  print('y_test shape:', y_test.shape)

with open(trainedModelName + '.pkl', mode='rb') as file:
   dlbac_alpha = pickle.load(file)

y_preds = dlbac_alpha.predict(x_test)
y_preds = (y_preds > 0.5).astype(int)

actual_acc = 0.0
for i in range(test_size):
    actual_acc = actual_acc + np.sum(y_test[i] == y_preds[i])

print('Accuracy Score:', actual_acc / (test_size * 4))


