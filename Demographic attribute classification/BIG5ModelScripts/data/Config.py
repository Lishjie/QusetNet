# coding:utf-8#

import os
import numpy as np
# get project absolute path
#project_path = os.path.abspath(__file__)
#project_path_ = project_path.split('/')
#project_path_ = project_path_[:-1]
#join_str = '/'
#project_path = join_str.join(project_path_)
#project_path = 'G:/Files/paper/r/classify/likertelmo-master'
features = np.load('/home/nlp/Desktop/likertelmo/BIG5ModelScripts/data/BIG5/features.npy')
project_path = '/home/nlp/Desktop/likertelmo/'

class Big5Config(object):
    # other
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 4
    keep_prob = 0.5
    total_num = 7099

    # dataset
    path = project_path+'/data/BIG5/'
    # absoluted path
    model_path = project_path+'data/BIG5/trained-model/model/'
    model_attention_path = project_path + \
        '/data/BIG5/trained-model/model-attention/'  # absoluted path
    question_var = 20
    num_steps = 56
    num_answer = 50
    num_feature = 7
    answer_dim = 5
    max_dim = 40 * answer_dim
    feature_dim = [14, 6, 4, 4, 4, 5, 160]

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
    learning_rate = 0.1 
    keep_prob = 0.5

    
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
    decay_steps = int(96867/batch_size)  
    decay_rate = 0.96    
