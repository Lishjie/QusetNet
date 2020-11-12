# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:16:59 2020

@author: Shijie Li
@description: 将隐层的拼接方式修改为对应单词的拼接
              {[hl0, hr2], [hl1, hr1], [hl2, hr0]}
"""
from BIG5ModelScripts import utils_classify as utils
import Config
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
sys.path.append('../../')
"""
@description: ELMO模型 + attention
@argvs:  config : 训练超参数和配置信息
         embedding_size: 第一个输入隐层的维度
         concat_mode : 1，2隐层之间的直连方式 
                      default 不进行直连 concat 拼接 add 相加的方式
         is_training : 是否为训练阶段(主要用于控制dropout)
         add_attention : 是否使用attention机制
"""


class Model(object):
    def __init__(self, config, feature_index, concat_mode='default', is_training=True, add_attention=False):
        # var
        self.concat_mode = concat_mode
        self.add_attention = add_attention
        self.config = config
        self.feature_index = feature_index  # 进行分类预测的问题
        self.embedding_size = self.config.embedding_size
        self.is_training = is_training
        self.num_answer = self.config.num_answer
        self.num_feature = self.config.num_feature
        self.answer_dim = self.config.answer_dim
        self.question_var = self.config.question_var
        self.batch_size = self.config.batch_size
        self.max_dim = self.config.max_dim
        self.max_grad_norm = self.config.max_grad_norm
        self.learning_rate = self.config.learning_rate
        self.total_dim = self.question_var * self.answer_dim
        # create placeholders
        self.ph_input_answer = tf.placeholder(
            tf.int32, shape=[None, self.num_answer])
        self.ph_input_answer_re = tf.placeholder(
            tf.int32, shape=[None, self.num_answer-1])
        self.ph_input_feature = tf.placeholder(
            tf.int32, shape=[None, self.num_feature-1, self.max_dim])
        self.ph_input_feature_re = tf.placeholder(
            tf.int32, shape=[None, self.num_feature, self.max_dim])
        self.ph_label_feature = tf.placeholder(tf.int32, shape=[None])

        # create graph
        self.create_model_graph()

    def create_feed_dict(self, a, b, c, d, e):
        feed_dict = {
            self.ph_input_answer: a,
            self.ph_input_answer_re: b,
            self.ph_input_feature: c,
            self.ph_input_feature_re: d,
            self.ph_label_feature: e
        }
        return feed_dict

    def attention(self, inputs, size, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # init [steps+1]
            init = tf.get_variable(str(size), shape=[size])
            alphas = tf.nn.softmax(init)
            # 复制attention的权重
            # alphas_tile [5, size]
            alphas_tile = tf.tile(tf.expand_dims(alphas, axis=0), [self.batch_size, 1])
            # output [5, 60]
#            output = tf.reduce_sum(inputs * tf.expand_dims(alphas_tile, -1), 1)
            output = inputs * tf.expand_dims(alphas_tile, -1)
        return output, alphas

    def build_rnn_graph_lstm(self, inputs, config, is_training, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.BasicLSTMCell(config.rnn_size, state_is_tuple=True)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            initState = cell.zero_state(self.batch_size, dtype=tf.float32)
            # outputs [5, 49, 150]
            outputs, final_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initState, dtype=tf.float32)
            # 根据参数选择隐层拼接的方式
            if self.concat_mode == 'concat':
                # 隐层拼接 [5, 49, 200]
                outputs = tf.concat([outputs, inputs], axis=2)
            elif self.concat_mode == 'add':
                # 隐层拼接 [5, 49, 150]
                outputs = outputs + inputs
        return outputs
    def create_model_graph(self):
        # 固定第一个隐层的输入 forwards
        self.f_input = tf.tile(tf.expand_dims(tf.cast(self.ph_input_answer, dtype=tf.float32), axis=-1),[1, 1, self.embedding_size])
        self.f_input = tf.divide(self.f_input, tf.sqrt(tf.cast(self.embedding_size, dtype=tf.float32)))
        # 删除将要进行预测的问题向量
        if self.feature_index != self.num_feature - 1:
            self.ph_input_feature_ = tf.concat([tf.slice(self.ph_input_feature, [0, 0, 0], [-1, self.feature_index, -1]),
                                                tf.slice(self.ph_input_feature, [0, self.feature_index+1, 0], [-1, -1, -1])],1)
        else:
            self.ph_input_feature_ = self.ph_input_feature
        self.f_input = tf.concat([self.f_input, tf.cast(self.ph_input_feature_, dtype=tf.float32)], 1)

        # 固定第一个隐层的输入 backwards
        self.b_input = tf.tile(tf.expand_dims(tf.cast(self.ph_input_answer_re, dtype=tf.float32), axis=-1),[1, 1, self.embedding_size])
        self.b_input = tf.divide(self.b_input, tf.sqrt(tf.cast(self.embedding_size, dtype=tf.float32)))
        # 删除将要进行预测的问题向量
        self.ph_input_feature_re_ = tf.concat([tf.slice(self.ph_input_feature_re, [0, 0, 0], [-1, self.num_feature - self.feature_index - 1, -1]),
                                               tf.slice(self.ph_input_feature_re, [0, self.num_feature - self.feature_index, 0], [-1, -1, -1])],1)
        self.b_input = tf.concat([tf.cast(self.ph_input_feature_re_, dtype=tf.float32), self.b_input], 1)

        # ELMO layers
        self.f_lstm_output = self.build_rnn_graph_lstm(self.f_input,
                                                       self.config,
                                                       self.is_training,
                                                       "forward")

        self.b_lstm_output = self.build_rnn_graph_lstm(self.b_input,
                                                       self.config,
                                                       self.is_training,
                                                       "backward")

        # attention layers
        self.fi, self.f_alpha = self.attention(
            self.f_lstm_output, self.f_lstm_output.shape[1].value, "forward")
        self.bi, self.b_alpha = self.attention(
            self.b_lstm_output, self.b_lstm_output.shape[1].value, "backward")
        
        """
        修改隐层的拼接方式，修改步骤如下：
        1. 将bi的数组倒置
        2. 将fi的数组最后的位置插入bi数组中的最后的元素
        3. 将bi的数组的第一的位置插入fi数组中的第1个元素
        4. 将两个数组进行拼接
        """
        # input
        self.bi_concat_input = tf.reverse(self.b_input, axis=[1])
        self.fi_concat_input = tf.concat((self.f_input, tf.slice(
            self.bi_concat_input, [0, 54, 0], [-1, 1, -1])), axis=1)
        self.bi_concat_input = tf.concat((tf.slice(self.fi_concat_input, [0, 0, 0], [-1, 1, -1]), self.bi_concat_input), axis=1)
        # lstm
        self.bi_concat_lstm = tf.reverse(self.b_lstm_output, axis=[1])
        self.fi_concat_lstm = tf.concat((self.f_lstm_output, tf.slice(
            self.bi_concat_lstm, [0, 54, 0], [-1, 1, -1])), axis=1)
        self.bi_concat_lstm = tf.concat((tf.slice(self.fi_concat_lstm, [0, 0, 0], [-1, 1, -1]), self.bi_concat_lstm), axis=1)
        # attention 
        self.bi_concat = tf.reverse(self.bi, axis=[1])
        self.fi_concat = tf.concat((self.fi, tf.slice(
            self.bi_concat, [0, 54, 0], [-1, 1, -1])), axis=1)
        self.bi_concat = tf.concat((tf.slice(self.fi_concat, [0, 0, 0], [-1, 1, -1]), self.bi_concat), axis=1)
        self.logit_concat = tf.concat((self.fi_concat, self.bi_concat,
                                       self.fi_concat_input, self.bi_concat_input,
                                       self.fi_concat_lstm, self.bi_concat_lstm), axis=2)
#        self.logit_concat = tf.concat((self.fi, self.bi), axis=1)
        self.logit_class = tf.layers.dense(tf.reduce_mean(
            self.logit_concat, axis=1), units=self.config.feature_dim[self.feature_index])
        self.logit_possible = tf.nn.softmax(self.logit_class)
        self.logit_predict = tf.cast(
            tf.argmax(self.logit_possible, 1), tf.int32)
        self.logit_label = tf.one_hot(
            self.ph_label_feature, depth=self.config.feature_dim[self.feature_index])
        self.accuracy_class = tf.reduce_mean(
            tf.cast(tf.equal(self.logit_predict, self.ph_label_feature), tf.float32))
        self.class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logit_class, labels=self.logit_label)
        self.class_loss_ = tf.reduce_sum(self.class_loss)

        # total loss
        self.cost = self.class_loss_

        if not self.is_training:
            return
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    # 通过控制台选择需要进行分类的任务index
    #    feature_index = sys.argv[1]
    feature_index = 0

    config = Config.Big5Config()
    config.batch_size = 5
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    config_tf.inter_op_parallelism_threads = 1
    config_tf.intra_op_parallelism_threads = 1
    data = utils.DataStream()
    a, b, c, d, e = data.mask_feed_dic(0, np.arange(7187), feature_index)

    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(config, feature_index, concat_mode='add',
                      is_training=True, add_attention=True)

        tf.global_variables_initializer().run()

        logit_concat, \
        b_input, \
        f_input, \
        bi_concat, \
        fi_concat, \
        logit_predict, \
        accuracy_class, \
        cost = session.run([m.logit_concat,
                            m.b_input,
                            m.f_input,
                            m.bi_concat,
                            m.fi_concat,
                            m.logit_predict,
                            m.accuracy_class,
                            m.cost], feed_dict=m.create_feed_dict(a, b, c, d, e))
