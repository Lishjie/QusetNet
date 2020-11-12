#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import Transformer_not_mul_mse_concat_classify as Model ###
import utils_classify as utils
import Config
# dont know what the number means

#import seaborn as sns
    
config = Config.Big5Config()
# config.keep_prob = 1.0###
# config.batch_size = 5###
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

arr = np.arange(87179)
feature_index = 6####

with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
    initializer = tf.random_uniform_initializer(-config.init_scale, \
                                                config.init_scale)
    
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtest = Model.Model(config, feature_index, 'add', False, False)###
    tf.global_variables_initializer().run()  
    model_saver = tf.train.Saver()
    print ('model loading ...')
    model_saver.restore(session, config.path+'trained-model/model-transformer-concat6/lm-6')
    print ('Done!')

    
    for ids in range(5749//mtest.batch_size, 7187//mtest.batch_size):
        if ids%256==0:####dimeension isnt correct
            print(ids,end=",")
        a,b,c,d,e = data.mask_feed_dic(ids*5,np.arange(96867),feature_index)####correct batch_size to number 5
        feed_dicts = mtest.create_feed_dict(a,b,c,d,e)
        # not add attention
#        f_logit, f_label, b_logit, b_label = session.run([mtest.f_logit_,
#                                                          mtest.f_label,
#                                                          mtest.b_logit_,
#                                                          mtest.b_label],feed_dicts)
        # add attention

###################################################
        accuracy_class = session.run([

                                      mtest.accuracy_class],feed_dicts)
        acc_list.append(accuracy_class)

# top_5 = alphas_blist.tolist()
# top_5_index = map(top_5.index, heapq.nlargest(5, top_5))
# print(list(top_5_index))
mse = np.mean(acc_list)
print(mse)
