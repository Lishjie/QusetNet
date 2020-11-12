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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from BIG5ModelScripts import utils_classify as utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import Config
import numpy as np
import sys
import time

sys.path.append('..')
sys.path.append('.')


config = Config.Big5Config()

feature_index = 0
encoder = OneHotEncoder()
answers = utils.answers
features = utils.features - 1
label = encoder.fit_transform(np.reshape(
    features[:, feature_index], (-1, 1))).toarray()
label_ = features[:, feature_index]
features = np.delete(features, feature_index, axis=1)
features = encoder.fit_transform(features).toarray()
data = np.concatenate((answers, features), axis=1)
test_scores =[]
print('start splitting')
print('the featue_index is ' + str(feature_index))

print('start training')
best_score = 0.000
for i in range(5):
    X_trainval, X_test, y_trainval, y_test = train_test_split(data, label_, test_size=0.1)
    X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,test_size=0.11)
#    print(X_train.shape[0],X_val.shape[0],X_test.shape[0])
    train_start_time = time.time()
    for gamma in [0.0001, 0.001,0.01,0.1,1,10]:
        for C in [2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),1,2,4,8,16,32]:
            svm_ = svm.SVC(kernel='rbf',gamma=gamma,C=C)
            svm_.fit(X_train,y_train)
            score_val = svm_.score(X_val,y_val)
            if score_val > best_score:
                best_score = score_val
                best_parameters = {'gamma':gamma,'C':C}
    svm_trainval = svm.SVC(**best_parameters) #使用最佳参数，构建新的模型
    
    select_best_para_time  = time.time()
    
    svm_trainval.fit(X_trainval,y_trainval) #使用训练集和验证集进行训练，more data always results in good performance.
    
    best_para_train_end_time  = time.time()
    test_score = svm_trainval.score(X_test,y_test) # evaluation模型评估
    test_time = time.time()
    test_scores.append(test_score)
    print('------------------------'+ str(i) + '-----------------------------------')
    print("Best score on validation set:{:.3f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))
    print("Best score on test set:{:.3f}".format(test_score))     
    print('gird search use time ' + str(select_best_para_time -train_start_time ))  
    
    print('trainval data train use time ' + str(best_para_train_end_time -select_best_para_time ))  
    
    print('test use time ' + str(test_time -best_para_train_end_time )) 
    print('all use time ' + str(test_time -train_start_time )) 

