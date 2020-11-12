# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:08:46 2019

@author: Ruby
"""

import itertools
import operator
import numpy as np
from collections import Counter
from random import sample

test_sample_num = 640


def int_unk(x):
    try:
        value_int = int(x)
    except:
        value_int = np.random.randint(0, 5)
    return value_int


def com_value_num(num, test_dict):  # 计算一个person选择的连续的5个选项中，出现频次为num的选项个数
    cont = Counter(list(test_dict.values()))[num]
    return cont


def com_one_person_item(row):  # row = list
    a = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    num_times = [(k, len(list(v))) for k, v in itertools.groupby(row)]

    num_times.sort(key=operator.itemgetter(1))
    c = {k: v for k, v in num_times}
    a.update(c)
    # print(a)

    return a  # 字典，未排序

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
dict_total_pos = {}
for i, row in enumerate(train_positive):
    temp_dict = com_one_person_item(row)
    X, Y = Counter(temp_dict), Counter(dict_total_pos)
    dict_total_pos = dict(X+Y)
    # print(dict_total_pos)

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
batches = []
# 准备训练集假样本
train_negative = []
for i in range(70000):
    row = np.random.randint(0, 5, 48).tolist()
    train_negative.append(row)
dict_total_neg = {}
for i, row in enumerate(train_negative):
    temp_dict = com_one_person_item(row)
    X, Y = Counter(temp_dict), Counter(dict_total_neg)
    dict_total_neg = dict(X+Y)
    # print(dict_total_neg)

    #train_sentences, train_labels =  load_train_data(train_fp, train_negative_sam)

for total_batch in range(0, 10001, 50):
    acc = 0

# 准备测试集
    test_fp = './save/riasec/real_data_test.txt'
    # 'samples_'+str(total_batch)+'.txt'
    '''
    test_positive=test_fp.readlines()
    test_positive_sampling = sample(test_positive,100) #sample
    test_positive_sam='test_sam/D_positive_eval_sample100_'+str(total_batch)+'.txt'
    with open(test_positive_sam,'w') as fout:
        for poem in test_positive_sampling:
            fout.write(poem)
    '''

    test_fn = './save/riasec/samples/samples_' + str(total_batch) + '.txt'
    #test_fn = './save/riasec/test_negative_random.txt'
    # 'result_J/G_negative_eval_1000_'+str(total_batch)+'.txt'
    #

    '''
    test_negative=test_fn.readlines()
    test_negative_sampling = sample(test_negative,100) #sample  
    test_negative_sam='test_sam/D_negative_eval_sample100_'+str(total_batch)+'.txt'
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

    for test_index, test_elm in enumerate(test_sentences):
        test_dict = com_one_person_item(test_elm)
        # print(test_dict)

        if (test_dict[0] >= 4 and com_value_num(2, test_dict) < 3)or test_dict[0] == 0 or (test_dict[1] >= 4 and com_value_num(2, test_dict) < 3)or test_dict[1] == 0 or (test_dict[2] >= 4 and com_value_num(2, test_dict) < 3) or test_dict[2] == 0 or (test_dict[3] >= 4 and com_value_num(2, test_dict) < 3) or test_dict[3] == 0 or (test_dict[4] >= 4 and com_value_num(2, test_dict) < 3)or test_dict[4] == 0:
            y_labels.insert(test_index, [0, 1])  # 也可以append顺序插入

        else:
            y_labels.insert(test_index, [1, 0])
        if y_labels[test_index][0] == test_labels[test_index][0]:
            cont += 1

    acc = cont/len(test_labels)

    #acc = metrics.r2_score(labels, y_labels,multioutput='raw_values')

    # print('acc:',acc)

    print(acc, ' ', total_batch)
    ACC.append(acc)
    batches.append(total_batch)

with open('./save/riasec/acc_result/longest_acc.txt', 'w') as f:
    for i in range(len(ACC)):
        f.write(str(ACC[i]) + ' ' + str(batches[i]) + '\n')

print(ACC)
