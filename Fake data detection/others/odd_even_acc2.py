# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:10:13 2019

@author: Ruby
"""

import numpy as np
from sklearn.model_selection import KFold
#from scipy.spatial.distance import pdist
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from random import sample
from sklearn import metrics
from sklearn.metrics import f1_score
test_sample_num = 12800


def int_unk(x):
    try:
        value_int = int(x)
    except:
        value_int = np.random.randint(0, 5)
    return value_int


def fill_ndarray(K_CORR):

    a = np.array(K_CORR)
    where_are_nan = np.isnan(a)
    where_are_inf = np.isinf(a)
    a[where_are_nan] = 0
    a[where_are_inf] = 0
    a_mean = np.mean(a)
    a[where_are_nan] = a_mean
    a[where_are_inf] = a_mean

    return a_mean


def Spearman(cor):  # Spearman-Brown校正相关系数的公式
    cor_res_rel = 2*cor/(1+cor)
    return cor_res_rel


def com_one_person(row):

    train_pos_E = row[0:10]
    train_pos_N = row[10:20]
    train_pos_A = row[20:30]
    train_pos_C = row[30:40]
    train_pos_O = row[40:50]
    train_pos_E_odd = train_pos_E[::2]
    train_pos_E_even = train_pos_E[1::2]
    train_pos_N_odd = train_pos_N[::2]
    train_pos_N_even = train_pos_N[1::2]
    train_pos_A_odd = train_pos_A[::2]
    train_pos_A_even = train_pos_A[1::2]
    train_pos_C_odd = train_pos_C[::2]
    train_pos_C_even = train_pos_C[1::2]
    train_pos_O_odd = train_pos_O[::2]
    train_pos_O_even = train_pos_O[1::2]
    odd_mean_list = []
    even_mean_list = []
    train_pos_E_odd_arr = np.array(train_pos_E_odd)
    train_pos_E_odd_mean = np.mean(train_pos_E_odd_arr)
    odd_mean_list.append(train_pos_E_odd_mean)
    train_pos_E_even_arr = np.array(train_pos_E_even)
    train_pos_E_even_mean = np.mean(train_pos_E_even_arr)
    even_mean_list.append(train_pos_E_even_mean)
    train_pos_N_odd_arr = np.array(train_pos_N_odd)
    train_pos_N_odd_mean = np.mean(train_pos_N_odd_arr)
    odd_mean_list.append(train_pos_N_odd_mean)
    train_pos_N_even_arr = np.array(train_pos_N_even)
    train_pos_N_even_mean = np.mean(train_pos_N_even_arr)
    even_mean_list.append(train_pos_N_even_mean)
    train_pos_A_odd_arr = np.array(train_pos_A_odd)
    train_pos_A_odd_mean = np.mean(train_pos_A_odd_arr)
    odd_mean_list.append(train_pos_A_odd_mean)
    train_pos_A_even_arr = np.array(train_pos_A_even)
    train_pos_A_even_mean = np.mean(train_pos_A_even_arr)
    even_mean_list.append(train_pos_A_even_mean)
    train_pos_C_odd_arr = np.array(train_pos_C_odd)
    train_pos_C_odd_mean = np.mean(train_pos_C_odd_arr)
    odd_mean_list.append(train_pos_C_odd_mean)
    train_pos_C_even_arr = np.array(train_pos_C_even)
    train_pos_C_even_mean = np.mean(train_pos_C_even_arr)
    even_mean_list.append(train_pos_C_even_mean)
    train_pos_O_odd_arr = np.array(train_pos_O_odd)
    train_pos_O_odd_mean = np.mean(train_pos_O_odd_arr)
    odd_mean_list.append(train_pos_O_odd_mean)
    train_pos_O_even_arr = np.array(train_pos_O_even)
    train_pos_O_even_mean = np.mean(train_pos_O_even_arr)
    even_mean_list.append(train_pos_O_even_mean)
    corr = np.corrcoef(np.array(odd_mean_list), np.array(even_mean_list))[0][1]
    corr_SB = Spearman(corr)

    return corr_SB


def com_one_person_allitem(row):
    pos_odd = row[::2]
    pos_even = row[1::2]
    pos_odd_arr = np.array(pos_odd)
    pos_even_arr = np.array(pos_even)
    corr = np.corrcoef(pos_odd_arr, pos_even_arr)[0][1]
    corr_SB = Spearman(corr)
    return corr


def con_all_person_mean(train_positive):
    CORR = []
    for i in range(len(train_positive)):
        corr = com_one_person_allitem(train_positive[i])
        CORR.append(corr)
    mean = fill_ndarray(CORR)
    return mean

# 真假样本混合打乱


def load_test_data(positive_file, negative_file):
    # Load data
    positive_examples = []
    negative_examples = []
    with open(positive_file)as fin:
        fin_list = []
        for line in fin:
            fin_list.append(line)
        for line in sample(fin_list, test_sample_num):
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            positive_examples.append(parse_line)
    with open(negative_file)as fin:
        fin_list = []
        for line in fin:
            fin_list.append(line)
        for line in sample(fin_list, test_sample_num):
            line = line.strip()
            line = line.split()
            parse_line = [int_unk(x) for x in line]
            if len(parse_line) == 48:
                negative_examples.append(parse_line)
    sentences = np.array(positive_examples + negative_examples)

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)

    # Shuffle the data
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    sentences = sentences[shuffle_indices]
    labels = labels[shuffle_indices]
    return sentences, labels


def load_train_pos(positive_file):
    positive_examples = []
    with open(positive_file)as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            positive_examples.append(parse_line)
    labels = [[1, 0] for _ in positive_examples]
    sentences = np.array(positive_examples)
    return sentences, labels


ACC = []
# FPR=[]
# TPR=[]
# 准备训练集真样本

train_fp = np.loadtxt('./save/riasec/real_data_train.txt')
train_positive = train_fp.tolist()
# train_positive=sorces_pre(train_positive)
train_positive = np.array(train_positive)
# train_positive=sorces_pre(train_positive)
train_pos_mean = con_all_person_mean(train_positive)
'''
    train_fp=open('real_data_train.txt') 
    train_positive=train_fp.readlines()
    train_positive_sampling = sample(train_positive,500) 
    train_positive_sam='result_Z5/real_data_train_sam.txt'
    with open(train_positive_sam,'w') as fout:
        for poem in train_positive_sampling:
            fout.write(poem)
    train_fp=np.loadtxt('result_Z5/real_data_train_sam.txt')     
    train_positive=train_fp.tolist()
    print(train_positive)
'''

# 准备训练集假样本
train_negative = []
for i in range(70000):
    row = np.random.randint(0, 5, 48).tolist()
    train_negative.append(row)
train_negative = np.array(train_negative)
# train_negative=sorces_pre(train_negative)
train_neg_mean = con_all_person_mean(train_negative)

'''
train_pos_array = np.array(train_positive)
train_pos_col_mean = np.sum(train_pos_array,axis=0)/len(train_pos_array)
train_neg_array = np.array(train_negative)
train_neg_col_mean = np.sum(train_neg_array,axis=0)/len(train_neg_array)
train_pos_E_mean, train_pos_N_mean, train_pos_A_mean, train_pos_C_mean, train_pos_O_mean = com_one_person(train_pos_col_mean)
train_neg_E_mean, train_neg_N_mean, train_neg_A_mean, train_neg_C_mean, train_neg_O_mean = com_one_person(train_neg_col_mean)    
'''
batches = []
for total_batch in range(0, 10001, 50):
    acc = 0
# 准备测试集
    test_fp = './save/riasec/real_data_test.txt'
# 'result_J/D_positive_eval_sample1000_'+str(total_batch)+'.txt'
    '''
    test_positive=test_fp.readlines()
    test_positive_sampling = sample(test_positive,100) #sample
    test_positive_sam='test_sam/D_positive_eval_sample100_'+str(total_batch)+'.txt'
    with open(test_positive_sam,'w') as fout:
        for poem in test_positive_sampling:
            fout.write(poem)
    '''
    test_fn = './save/riasec/samples/samples_' + str(total_batch) + '.txt'
    # 'result_J/G_negative_eval_1000_'+str(total_batch)+'.txt'
    # 'test_negative_random.txt'

    '''
    test_negative=test_fn.readlines()
    test_negative_sampling = sample(test_negative,100) #sample  
    test_negative_sam='test_sam/G_negative_eval_sample100_'+str(total_batch)+'.txt'
    with open(test_negative_sam,'w') as fout:
        for poem in test_negative_sampling:
            fout.write(poem)
    '''
    test_sentences, test_labels = load_test_data(test_fp, test_fn)  # 打乱测试集
    #test_sentences, test_labels =  load_train_pos(test_fn)

    cont = 0
    # fp=0
    # tp=0
    y_labels = []
    # test_sentences=sorces_pre(test_sentences)
    for test_index, test_elm in enumerate(test_sentences):
        test_corr = com_one_person(test_elm)
        pos_dis = abs(test_corr-train_pos_mean)
        neg_dis = abs(test_corr-train_neg_mean)
        if pos_dis < neg_dis:
            y_labels.insert(test_index, [0, 1])  # 也可以append顺序插入

        else:
            y_labels.insert(test_index, [1, 0])
        if y_labels[test_index][0] == test_labels[test_index][0]:
            cont += 1

    acc = cont/len(test_labels)

    #acc = metrics.r2_score(labels, y_labels,multioutput='raw_values')

    print('acc:', acc, ' ', total_batch)
    ACC.append(acc)
    batches.append(total_batch)
with open('./save/riasec/acc_result/odd_acc.txt', 'w') as f:
    for i in range(len(ACC)):
        f.write(str(ACC[i]) + ' ' + str(batches[i]) + '\n')
print(ACC)
