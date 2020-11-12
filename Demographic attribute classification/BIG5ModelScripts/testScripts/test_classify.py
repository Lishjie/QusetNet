#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:38:18 2019

@author:      lishijie
@description: 使用序损失模型的测试文件
"""
import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
from model import ELMO_not_mul_mse_concat_classify as Model
from BIG5ModelScripts import utils_classify as utils
import Config
import matplotlib.pyplot as plt
import heapq
#import seaborn as sns
    
config = Config.Big5Config()
config.keep_prob = 1.0
config.batch_size = 5
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1
data = utils.DataStream()

#list_forwards_attentions = []
#list_backwards_attentions = []
#f_logit_mse = []
#f_label_mse = []
#b_logit_mse = []
#b_label_mse = []
alphas_flist_list = []
acc_list = []

total = 7099
arr = np.arange(total)
np.random.shuffle(arr)
arr_train = arr[0:int(0.8*total)].copy()
arr_test = arr[int(0.8*total):int(0.9*total)].copy()
feature_index = 0

with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
    initializer = tf.random_uniform_initializer(-config.init_scale, \
                                                config.init_scale)
    
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtest = Model.Model(config, feature_index, 'add', False, True)
    tf.global_variables_initializer().run()  
    model_saver = tf.train.Saver()
    print ('model loading ...')
    model_saver.restore(session, '../../data/BIG5/trained-model/model-classify-concat0/lm-332')
    print ('Done!')

    
    for ids in range(5749//mtest.batch_size, 7187//mtest.batch_size):
        if ids%200==0:
            print(ids,end=",")
        a,b,c,d,e = data.mask_feed_dic(ids*config.batch_size,np.arange(96867),feature_index)
        feed_dicts = mtest.create_feed_dict(a,b,c,d,e)
        # not add attention
#        f_logit, f_label, b_logit, b_label = session.run([mtest.f_logit_,
#                                                          mtest.f_label,
#                                                          mtest.b_logit_,
#                                                          mtest.b_label],feed_dicts)
        # add attention
        alphas_flist, \
        alphas_blist, \
        accuracy_class = session.run([mtest.f_alpha,
                                      mtest.b_alpha,
                                      mtest.accuracy_class],feed_dicts)
        acc_list.append(accuracy_class)

top_5 = alphas_flist.tolist()
top_5_index = map(top_5.index, heapq.nlargest(5, top_5))
print(list(top_5_index))
mse = np.mean(acc_list)
#num_list = alphas_flist.tolist()
#plt.figure(figsize=(50,50))
#plt.bar(range(len(num_list)), num_list, facecolor = 'lightskyblue', edgecolor = 'white')
#plt.show() 
# 开始遍历保存前向attention的结果
#for row, alphas_flist_row in enumerate(alphas_flist): 
#    for col, alphas_flist_col in enumerate(alphas_flist[row]):
#        f_attention_array[col, row+1] = alphas_flist_col
#
#f, (ax1) = plt.subplots(figsize=(50,50),nrows=1)
#sns.heatmap(f_attention_array, annot=True, ax=ax1)
        
#    f_logit_list_int = np.array(np.reshape(np.rint(f_logit_list), [-1]), dtype=np.int)
#    b_logit_list_int = np.array(np.reshape(np.rint(b_logit_list), [-1]), dtype=np.int)
#    
#    # ji suan zheng xiang de zheng que lv
#    f_accuracy = np.mean(np.equal(np.array(np.reshape(f_label_list, [-1]),\
#                                           dtype=np.int), f_logit_list_int))
#    # ji suan fan xiang de zheng que lv
#    b_accuracy = np.mean(np.equal(np.array(np.reshape(b_label_list, [-1]),\
#                                           dtype=np.int), b_logit_list_int))
    