#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:34:04 2019

@author: lishijie
@description: use logistic to classify
"""
import numpy as np
import Config
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from BIG5ModelScripts import utils_classify as utils
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sys
sys.path.append('..')
sys.path.append('/home/nlp/Desktop/likertelmo/')
#from sklearn.grid_search import GridSearchCV
sys.path.append('..')
sys.path.append('.')


config = Config.Big5Config()

Acc = []
# for ii in [0]:
#for ii in [0, 1, 3, 4, 5]:
for ii in [5]:
    feature_index = ii
    encoder = OneHotEncoder()
    answers = utils.answers
    features = utils.features - 1
    label = encoder.fit_transform(np.reshape(
        features[:, feature_index], (-1, 1))).toarray()
    label_ = features[:, feature_index]
    features = np.delete(features, feature_index, axis=1)
    features = encoder.fit_transform(features).toarray()
    data = np.concatenate((answers, features), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        data, label_, test_size=0.2)
    print(X_train.shape)

    # logistic
    # train
    #data = utils.DataStream(utils.answers, utils.features, config, config.batch_size)
    #samples, target = data.tran_feed_dic(0, 77493 ,feature_index)
    #samples_answer = samples[:, 0:48]
    #samples_feature = samples[:, 48:]
    #samples_feature_ = encoder.fit_transform(samples_feature).toarray()
    #samples_ = np.concatenate((samples_answer, samples_feature_), axis=1)

    #classifier = LogisticRegression()
    #classifier.fit(X_train, y_train)

    # predicts
    #X, y = data.tran_feed_dic(77493, 87179, feature_index)
    #X_answer = X[:, 0:48]
    #X_feature = X[:, 48:]
    #X_feature_ = encoder.fit_transform(X_feature).toarray()
    #X_ = np.concatenate((X_answer, X_feature_), axis=1)

    #accs_logistic = classifier.score(X_test, y_test)

    # random forest
    # acc = 0
    # for i in range(10):
    #     # train
    #     clf = RandomForestClassifier(n_estimators=300)
    #     clf.fit(X_train, y_train)

    #     # predict
    #     y1_ = clf.predict(X_test)
    #     accs_randomforest = clf.score(X_test, y_test)
    #     print(accs_randomforest)
    #     acc = acc+accs_randomforest
    # print(acc/10)
    # Acc.append(acc/10)
    # print('iiiiii:', ii)
# print(Acc)
# svm
# train
#svm_ = svm.SVC(kernel='rbf')
#svm_.fit(X_train, y_train)

# predict
#accs_svm = svm_.score(X_test, y_test)
