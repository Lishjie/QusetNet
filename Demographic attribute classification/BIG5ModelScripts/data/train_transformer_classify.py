#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import utils_classify as utils
import Transformer_not_mul_mse_concat_classify as Model
import Config
import time
import numpy as np
import tensorflow as tf
import math

config = Config.Big5Config()
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1
data = utils.DataStream()

total = config.total_num
arr = np.arange(total)
np.random.shuffle(arr)
arr_train = arr[0:int(0.8*total)].copy()
arr_test = arr[int(0.8*total):].copy()


def run_epoch(session, m, feature_index):
    """Runs the model on the given data."""
    epoch_size =int(0.8*total)//m.batch_size
    start_time = time.time()
    np.random.shuffle(arr_train)
#    f_acc = 0.0
#    b_acc = 0.0

    accuracy_list = []
    for step in range(epoch_size):
        a, b, c, d, e = data.mask_feed_dic(step*config.batch_size, arr_train, feature_index)
        feed_dicts = m.create_feed_dict(a, b, c, d, e)

        accuracy,_, cost = session.run([m.accuracy_class, m.train_op,m.cost], feed_dicts)
        accuracy_list.append(accuracy)

        if step and step % (epoch_size // 10) == 0:
            print("%.2f cost: %.3f accuracy: %.4f cost-time: %.2f s\n" %(step*1.0 / epoch_size, cost, np.mean(accuracy_list),(time.time() - start_time)))
            start_time = time.time()
            accuracy_list = []
    return cost


def main(_):
    #    feature_index = sys.argv[1]
    feature_index = 6
    feature_index = int(feature_index)
#    best_f_acc = 0.0  # ji lu qian xiang de zui gao jing du
#    best_b_acc = 0.0  # ji lu hou xiang de zui gao jing du

    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        # try1: change the initializer to xavier
        # initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model.Model(config, feature_index, concat_mode='add', is_training=True, add_attention=True)
        
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mValidate = Model.Model(config, feature_index, concat_mode='add',is_training=False,add_attention=True)

        tf.global_variables_initializer().run()

        model_saver = tf.train.Saver(tf.global_variables())

        max_accuracy = 0.0

        for i in range(config.iteration):
            print("Training Epoch: %d ..." % (i+1))
            train_perplexity = run_epoch(session, m, feature_index)
            print("Epoch: %d Train Perplexity: %.3f" %(i + 1, train_perplexity))

            accuracys_list = []

            for ids in range((config.total_num - int(0.8*config.total_num))//mValidate.batch_size):
                a, b, c, d, e = data.mask_feed_dic(ids*config.batch_size, arr_test, feature_index)
                feed_dicts = mValidate.create_feed_dict(a, b, c, d, e)

                accuracy = session.run([mValidate.accuracy_class], feed_dicts)

                accuracys_list.append(accuracy)

            if max_accuracy < np.mean(accuracys_list):
                max_accuracy = np.mean(accuracys_list)
                print('model saving ... current Verification accuracy: %.4f\n'% (max_accuracy))
                model_saver.save(session, config.path+'trained-model/model-transformer-concat'+str(feature_index)+'/'+'lm-%d' % (i+1))
                print('Done!')
            else:
                print('current accuracy: ', np.mean(accuracys_list), '\n')

if __name__ == "__main__":
    tf.app.run()