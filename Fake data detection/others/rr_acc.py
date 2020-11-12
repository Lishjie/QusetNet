# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:38:16 2019

@author: Ruby
"""

import numpy as np
#import random
from random import sample

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


def response_reliability(train_positive):
    item_pairs = []
    used = []
    for i in range(0, len(train_positive[0])-1):
        ed = 0
        if i not in used:
            items_euclidean_dis = []
            for j in range(i+1, len(train_positive[0])):
                if j not in used:
                    temp = np.sqrt(
                        np.sum((train_positive[:, i] - train_positive[:, j]) ** 2))

                    if temp < ed or ed == 0:
                        ed = temp
                        min_dex = j

            used.append(min_dex)
            item_pair = (i, min_dex)
            item_pairs.append(item_pair)
    pos_one = [x[0] for x in item_pairs]
    pos_two = [x[1] for x in item_pairs]
    return pos_one, pos_two


def com_cor(row, pos_one, pos_two):
    row_one = [row[i] for i in pos_one]
    row_two = [row[i] for i in pos_two]
    corr = np.corrcoef(np.array(row_one), np.array(row_two))[0][1]
    corr_SB = Spearman(corr)
    return corr_SB


def com_cor_res_rel(row):
    row_array = np.array(row)
    np.random.seed(2)
    index_1 = np.random.choice(row_array.shape[0], 25, replace=False)
    row_array_1 = row_array[index_1]
    index_2 = np.array(range(row_array.shape[0]))
    index_2 = np.delete(index_2, index_1)
    row_array_2 = row_array[index_2]
    cor = np.corrcoef(row_array_1, row_array_2)[0][1]

    # random.seed(10)
    #cor = np.corrcoef(np.array(random.sample(row,25)),np.array(row[5:10]+row[15:20]+row[25:30]+row[35:40]+row[45:50]))[0][1]
    cor_res_rel = 2*cor/(1+cor)
    # print(cor)
    # print(cor_res_rel)
    return cor_res_rel


def con_all_person_mean(train_positive, train_pos_one, train_pos_two):
    CORR = []
    for i in range(len(train_positive)):
        corr = com_cor(train_positive[i], train_pos_one, train_pos_two)
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

train_fp = np.loadtxt(
    './save/riasec/real_data_train.txt')
train_positive = train_fp.tolist()
fp_pos_one, fp_pos_two = response_reliability(np.array(train_positive))
fp_mean = con_all_person_mean(train_positive, fp_pos_one, fp_pos_two)
'''
cor_res_rel_total_pos = []
for idx, row in enumerate(train_positive):
    temp_cor_res_rel = com_cor_res_rel(row)
    cor_res_rel_total_pos.append(temp_cor_res_rel)
train_pos_cor_mean = np.mean(np.array(cor_res_rel_total_pos))
print(train_pos_cor_mean)
'''
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
fn_pos_one, fn_pos_two = response_reliability(np.array(train_negative))
fn_mean = con_all_person_mean(train_negative, fn_pos_one, fn_pos_two)
'''    
cor_res_rel_total_neg = []
for idx, row in enumerate(train_negative):
    temp_cor_res_rel = com_cor_res_rel(row)
    cor_res_rel_total_neg.append(temp_cor_res_rel)
train_neg_cor_mean = np.mean(np.array(cor_res_rel_total_neg))
print(train_neg_cor_mean)
'''
#train_sentences, train_labels =  load_train_data(train_fp, train_negative_sam)

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

    test_fn = './save/riasec/samples/samples_' + \
        str(total_batch) + '.txt'
    # 'result_J/G_negative_eval_1000_'+str(total_batch)+'.txt'
    # 'test_negative_random.txt'

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
        test_fp_corr = com_cor(test_elm, fp_pos_one, fp_pos_two)
        test_fn_corr = com_cor(test_elm, fn_pos_one, fn_pos_two)

        if abs(test_fp_corr-fp_mean) <= abs(test_fn_corr-fn_mean):
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
with open('./save/riasec/acc_result/rr_acc.txt', 'w') as f:
    for i in range(len(ACC)):
        f.write(str(ACC[i]) + ' ' + str(batches[i]) + '\n')


'''
            if y_labels[index][0] == 0 and labels[index][0] == 0:#真正
                fp += 1
                fpr = fp/len(positive_file)
                FPR.append(fpr)
            if y_labels[index][0] == 0 and  labels[index][0] == 1: #假正
                tp += 1
                tpr = tp/len(negative_file)
                TPR.append(tpr)
'''

'''
roc_auc = auc(FPR,TPR)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Percentage of simulated responses:50%')
plt.legend(loc="lower right")
plt.show()
'''
print(ACC)
