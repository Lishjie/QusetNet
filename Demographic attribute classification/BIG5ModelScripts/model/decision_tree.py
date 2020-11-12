#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:34:04 2019

@author: lishijie
@description: use logistic to classify
"""
import sys
sys.path.append('..')
sys.path.append('/home/nlp/Desktop/likertelmo/')
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from BIG5ModelScripts import utils_classify as utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import Config
import numpy as np
import sys
import time

sys.path.append('..')
sys.path.append('.')


config = Config.Big5Config()

feature_index = 6
encoder = OneHotEncoder()
answers = utils.answers
features = utils.features - 1
label = encoder.fit_transform(np.reshape(
    features[:, feature_index], (-1, 1))).toarray()
label_ = features[:, feature_index]
features = np.delete(features, feature_index, axis=1)
features = encoder.fit_transform(features).toarray()
data = np.concatenate((answers, features), axis=1)

print('start splitting')
print('the featue_index is ' + str(feature_index))

print('start training')

X_trainval, X_test, y_trainval, y_test = train_test_split(data, label_, test_size=0.1)
X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,test_size=0.11)

clf = DecisionTreeClassifier(random_state=0)
print(cross_val_score(clf, data, label_, cv=10))
