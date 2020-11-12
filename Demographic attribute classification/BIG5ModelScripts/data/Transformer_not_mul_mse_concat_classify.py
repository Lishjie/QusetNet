# -*- coding: utf-8 -*-

import utils_classify as utils
import Config
import numpy as np
import tensorflow as tf
import sys
from multihead import *
from model_helper import *
sys.path.append('..')



class Model(object):
    def __init__(self, config, feature_index, concat_mode='default', is_training=True, add_attention=False):
        # var
        self.concat_mode = concat_mode
        self.add_attention = add_attention
        self.config = config
        self.hidden_size = 64
        self.max_len = 32
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
            self.ph_input_answer_re: b,#带有re也不明白是什么意思  答案 特色 特色标签
            self.ph_input_feature: c,
            self.ph_input_feature_re: d,
            self.ph_label_feature: e
        }
        return feed_dict

#     def attention(self, inputs, size, scope_name):  #注意力机制部分
#         with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
#             # init [steps+1]
#             init = tf.get_variable(str(size), shape=[size])
#             alphas = tf.nn.softmax(init)
#             # 复制attention的权重
#             # alphas_tile [5, size]
#             alphas_tile = tf.tile(tf.expand_dims(alphas, axis=0), [self.batch_size, 1])
#             # output [5, 60]
# #            output = tf.reduce_sum(inputs * tf.expand_dims(alphas_tile, -1), 1)
#             output = inputs * tf.expand_dims(alphas_tile, -1)
#         return output, alphas

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


    def build_graph(self):
        print("building graph...")
        # embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                              trainable=True)
        # batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        # print(batch_embedded.shape)
        # multi-head attention
        # ma = multihead_attention(queries=batch_embedded, keys=batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        # outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        # outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        # logits = tf.layers.dense(outputs, units=self.n_class)

        # self.losses = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        # self.predictions = tf.argmax(tf.nn.softmax(logits), 1)
        # self.real = tf.argmax(self.label, axis=1, name="real_label")
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, self.real)
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        #
        # # optimization
        # loss_to_minimize = self.losses
        # tvars = tf.trainable_variables()
        # gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        # grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)
        #
        # self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
        #                                                name='train_step')
        print("graph built successfully!")
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

        ma = multihead_attention(queries=self.b_input, keys=self.b_input)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        # outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        # logits = tf.layers.dense(outputs, units=self.n_class)
        # ELMO layers
        # self.f_lstm_output = self.build_rnn_graph_lstm(self.f_input,
        #                                                self.config,
        #                                                self.is_training,
        #                                                "forward")
        #
        # self.b_lstm_output = self.build_rnn_graph_lstm(self.b_input,
        #                                                self.config,
        #                                                self.is_training,
        #                                                "backward")

        # attention layers
        # self.fi, self.f_alpha = self.attention(
        #     self.f_lstm_output, self.f_lstm_output.shape[1].value, "forward")
        # self.bi, self.b_alpha = self.attention(
        #     self.b_lstm_output, self.b_lstm_output.shape[1].value, "backward")

        # self.logit_concat = tf.concat((self.fi, self.bi), axis=1)
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


        bi, \
        logit_predict, \
        accuracy_class, \
        cost = session.run([
                            m.b_input,
                            m.logit_predict,
                            m.accuracy_class,
                            m.cost], feed_dict=m.create_feed_dict(a, b, c, d, e))
