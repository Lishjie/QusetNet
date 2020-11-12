# coding:utf-8#

import os
import numpy as np
# get project absolute path
project_path = os.path.abspath(__file__)
project_path_ = project_path.split('/')
project_path_ = project_path_[:-1]
join_str = '/'
project_path = join_str.join(project_path_)
features = np.load(project_path+'/data/RIASEC/features.npy')


class Big5Config(object):
    # other
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 4
    keep_prob = 0.5

    # dataset
    path = project_path+'/data/BIG5/'
    # absoluted path
    model_path = project_path+'/data/BIG5/trained-model/model/'
    model_attention_path = project_path + \
        '/data/BIG5/trained-model/model-attention/'  # absoluted path
    question_var = 20
    num_steps = 56
    num_answer = 50
    num_feature = 7
    answer_dim = 5
    max_dim = 6 * answer_dim
    feature_dim = [14, 5, 3, 4, 4, 5, 25]

    # train
    iteration = 100
    batch_size = 8
    num_layers = 2
    save_freq = 2
    rnn_size = max_dim
    embedding_size = max_dim


class RiaseConfig(object):
    # other
    init_scale = 0.04
    learning_rate = 0.1  # 初始学习�?    max_grad_norm = 15
    keep_prob = 0.5

    # 用于grid search的参数列�?    learning_rate_list = [0.1, 0.05]  # [0.01, 0.05, 0.1]
    keep_prob_list = [0.5, 0.7]  # [0.5, 0.7, 0.9]
    batch_size_list = [10, 50]  # [10, 30, 50]
    num_layers_list = [2]  # [2, 3, 4]

    # dataset
    path = project_path+'/data/RIASEC/'
    model_path = project_path+'/data/RIASEC/trained-model/mseModel/'
    question_var = 20
    num_steps = 72
    num_answer = 48
    num_feature = 25
    answer_dim = 5
    max_dim = 12 * answer_dim
    feature_dim = [7,  7,  7,  7,  7,
                   7,  7,  7,  7,  7,
                   4,  3,  3,  2,  5,
                   3,  12, 5,  5,  2,
                   4,  11, 2,  56, 3]
    feature_dim_re = [56, 2, 11, 4, 2,
                      5,  5, 12, 3, 5,
                      2,  3, 3,  4, 7,
                      7,  7, 7,  7, 7,
                      7,  7, 7,  7, 3]

    # train
    iteration_attention = 10000
    iteration = 10000
    batch_size = 10
    num_layers = 2
    save_freq = 2
    rnn_size = max_dim
    embedding_size = max_dim
    decay_steps = int(96867/batch_size)   # 衰减速度
    decay_rate = 0.96    # 衰减系数
