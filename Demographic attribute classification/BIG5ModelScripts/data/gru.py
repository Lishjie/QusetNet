# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:35:21 2019

@author: lishijie
@description: ELMO 对目录型变量进行分类任务
              包含将第1、2隐层输出进行拼接的操作
"""
import utils_classify as utils
import Config
import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
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
        self.keep_prob = 0.5
        self.rnn_dims = [256]
        self.rnn_size = 256
        self.fc_size = 500
        self.feature_index = feature_index  # 进行分类预测的问题
        self.embedding_size = self.config.embedding_size#词嵌入
        self.is_training = is_training
        self.num_answer = self.config.num_answer
        self.num_feature = self.config.num_feature
        self.answer_dim = self.config.answer_dim
        self.question_var = self.config.question_var#这个不太明白
        self.batch_size = self.config.batch_size
        self.max_dim = self.config.max_dim
        self.max_grad_norm = self.config.max_grad_norm
        self.learning_rate = self.config.learning_rate
        self.total_dim = self.question_var * self.answer_dim
        # create placeholders
        self.ph_input_answer = tf.placeholder( #输入答案
            tf.int32, shape=[None, self.num_answer])
        self.ph_input_answer_re = tf.placeholder( #输入答案倒过来
            tf.int32, shape=[None, self.num_answer-1])
        self.ph_input_feature = tf.placeholder( #输入特征
            tf.int32, shape=[None, self.num_feature-1, self.max_dim])
        self.ph_input_feature_re = tf.placeholder( #输入特征倒过来
            tf.int32, shape=[None, self.num_feature, self.max_dim])
        self.ph_label_feature = tf.placeholder(tf.int32, shape=[None]) #标签特征

        # create graph
        self.create_model_graph()

    def create_feed_dict(self, a, b, c, d, e):
        feed_dict = {
            self.ph_input_answer: a,
            self.ph_input_answer_re: b,#带有re也不明白是什么意思  答案 特色 特色标签
            self.ph_input_feature: c,
            self.ph_input_feature_re: d,
            self.ph_label_feature: e
        }
        return feed_dict

    def attention(self, inputs, size, scope_name):  #注意力机制部分
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

    def rnn_layer(self, embed):
        """
        RNN层
        """
        with tf.name_scope("rnn_layer"):
            embed = tf.nn.dropout(embed, keep_prob=self.keep_prob)  # dropout
            # --- 可选的RNN单元
            # tf.contrib.rnn.BasicRNNCell(size)
            # tf.contrib.rnn.BasicLSTMCell(size)
            # tf.contrib.rnn.LSTMCell(size)
            # tf.contrib.rnn.GRUCell(size, activation=tf.nn.relu)
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
            # tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(size)


            lstms = [tf.contrib.rnn.GRUCell(size, activation=tf.nn.relu) for size in self.rnn_dims]
            drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms]
            cell = tf.contrib.rnn.MultiRNNCell(drops)  # 组合多个 LSTM 层
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
            # lstm_outputs -> batch_size * max_len * n_hidden

            # bilstm
            # lstms_l = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.rnn_dims]
            # lstms_r = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.rnn_dims]
            # drops_l = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms_l]
            # drops_r = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms_r]
            # cell_l = tf.contrib.rnn.MultiRNNCell(drops_l)
            # cell_r = tf.contrib.rnn.MultiRNNCell(drops_r)

            # outputs, _ = tf.nn.bidirectional_dynamic_rnn(  # 双向LSTM
            #     cell_l,  # 正向LSTM单元
            #     cell_r,  # 反向LSTM单元
            #     inputs=embed,
            #     dtype=tf.float32,
            # )  # outputs -> batch_size * max_len * n_hidden; state(最终状态，为h和c的tuple) -> batch_size * n_hidden
            # lstm_outputs = tf.concat(outputs, -1)  # 合并双向LSTM的结果
            # outputs = lstm_outputs[:, -1]  # 返回每条数据的最后输出
            # logit_concat = tf.concat((lstm_outputs_l, lstm_outputs_r), axis=1)
        return lstm_outputs

    def fc_layer(self, inputs):
        """
        全连接层
        """
        # initializer = tf.contrib.layers.xavier_initializer()  # xavier参数初始化，暂没用到
        with tf.name_scope("fc_layer"):
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob, name='drop_out')  # dropout
            # outputs = tf.contrib.layers.fully_connected(inputs, self.fc_size, activation_fn=tf.nn.relu)
            outputs = tf.layers.dense(inputs, self.fc_size, activation=tf.nn.relu)
        return outputs

    # def output_layer(self, inputs):
    #     """
    #     输出层
    #     """
    #     with tf.name_scope("output_layer"):
    #         inputs = tf.layers.dropout(inputs, rate=1-self.keep_prob)
    #         outputs = tf.layers.dense(inputs, self.class_num, activation=None)
    #         # outputs = tf.contrib.layers.fully_connected(inputs, self.class_num, activation_fn=None)
    #     return outputs
    # def build_rnn_graph_lstm(self, inputs, config, is_training, scope_name):
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         cell = tf.contrib.rnn.BasicLSTMCell(config.rnn_size, state_is_tuple=True)
    #         if is_training and config.keep_prob < 1:
    #             cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    #         initState = cell.zero_state(self.batch_size, dtype=tf.float32)
    #         # outputs [5, 49, 150]
    #         outputs, final_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initState, dtype=tf.float32)
    #         # 根据参数选择隐层拼接的方式
    #         if self.concat_mode == 'concat':
    #             # 隐层拼接 [5, 49, 200]
    #             outputs = tf.concat([outputs, inputs], axis=2)
    #         elif self.concat_mode == 'add':
    #             # 隐层拼接 [5, 49, 150]
    #             outputs = outputs + inputs
    #     return outputs
    def create_model_graph(self):

        # # 固定第一个隐层的输入 forwards
        # self.f_input = tf.tile(tf.expand_dims(tf.cast(self.ph_input_answer, dtype=tf.float32), axis=-1),[1, 1, self.embedding_size])
        # self.f_input = tf.divide(self.f_input, tf.sqrt(tf.cast(self.embedding_size, dtype=tf.float32)))
        # # 删除将要进行预测的问题向量
        # if self.feature_index != self.num_feature - 1:
        #     self.ph_input_feature_ = tf.concat([tf.slice(self.ph_input_feature, [0, 0, 0], [-1, self.feature_index, -1]),
        #                                         tf.slice(self.ph_input_feature, [0, self.feature_index+1, 0], [-1, -1, -1])],1)
        # else:
        #     self.ph_input_feature_ = self.ph_input_feature
        # self.f_input = tf.concat([self.f_input, tf.cast(self.ph_input_feature_, dtype=tf.float32)], 1)

        # 固定第一个隐层的输入 backwards
        self.b_input = tf.tile(tf.expand_dims(tf.cast(self.ph_input_answer_re, dtype=tf.float32), axis=-1),[1, 1, self.embedding_size])
        self.b_input = tf.divide(self.b_input, tf.sqrt(tf.cast(self.embedding_size, dtype=tf.float32)))
        # 删除将要进行预测的问题向量
        self.ph_input_feature_re_ = tf.concat([tf.slice(self.ph_input_feature_re, [0, 0, 0], [-1, self.num_feature - self.feature_index - 1, -1]),
                                               tf.slice(self.ph_input_feature_re, [0, self.num_feature - self.feature_index, 0], [-1, -1, -1])],1)
        self.b_input = tf.concat([tf.cast(self.ph_input_feature_re_, dtype=tf.float32), self.b_input], 1)

        cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        initState = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_states = tf.nn.dynamic_rnn(cell, self.b_input, initial_state=initState, dtype=tf.float32)
        outputs = tf.concat([outputs, self.b_input], axis=2)

        # fc_outputs = self.fc_layer(rnn_outputs)
        # self.predictions = self.output_layer(fc_outputs)
        # # ELMO layers
        # self.f_lstm_output = self.build_rnn_graph_lstm(self.f_input,
        #                                                self.config,
        #                                                self.is_training,
        #                                                "forward")
        #
        # self.b_lstm_output = self.build_rnn_graph_lstm(self.b_input,
        #                                                self.config,
        #                                                self.is_training,
        #                                                "backward")

        # # attention layers
        # self.fi, self.f_alpha = self.attention(
        #     self.f_lstm_output, self.f_lstm_output.shape[1].value, "forward")
        # self.bi, self.b_alpha = self.attention(
        #     self.b_lstm_output, self.b_lstm_output.shape[1].value, "backward")
        #
        # self.logit_concat = tf.concat((rnn_outputs_f, rnn_outputs_b), axis=1)
        self.logit_class = tf.layers.dense(tf.reduce_mean(
            outputs, axis=1), units=self.config.feature_dim[self.feature_index])
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


    # feature_index = sys.argv[1]
    feature_index = 3

    config = Config.Big5Config()
    config.batch_size = 5
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    config_tf.inter_op_parallelism_threads = 1 #TensorFlow会根据 intra_op_parallelism_threads 参数的设置，将待实现的任务分配给各个线程
    config_tf.intra_op_parallelism_threads = 1 
    data = utils.DataStream()
    a, b, c, d, e = data.mask_feed_dic(0, np.arange(config.total_num), feature_index)

    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(config, feature_index, concat_mode='add',
                      is_training=True, add_attention=True)

        tf.global_variables_initializer().run()

        fi, \
        bi, \
        logit_predict, \
        accuracy_class, \
        cost = session.run([m.f_input,
                            m.b_input,
                            m.logit_predict,
                            m.accuracy_class,
                            m.cost], feed_dict=m.create_feed_dict(a, b, c, d, e))
