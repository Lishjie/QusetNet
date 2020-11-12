# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:35:08 2019

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
import math

import pandas as pd

test_sample_num = 12800

def int_unk(x):
    try:
        value_int = int(x)
    except:
        value_int = np.random.randint(0, 5)
    return value_int

def cor_compute(num,top_c,train_pos):
    
    
    item=48
    
    col_size = train_pos.shape
    print(col_size)
    cm = np.zeros((item,item))

    for i in range(col_size[1]):
        for j in range(i+1,item):
            c = np.corrcoef(train_pos[:,i],train_pos[:,j])[0][1]        
        
            cm[i][j] = c

    cm_re = cm.reshape(1,item*item)

    cm_re_squ = np.squeeze(cm_re)
    print(cm_re_squ.shape)
    top_idx = np.argsort(num*cm_re_squ)[0:top_c]
    elm = []
    cor_value=[]
    for i in range(top_c):
        row = math.floor(top_idx[i]/item)
        col = top_idx[i]%item
        elm.append((row,col))
        cor_value.append(cm[row][col])
    mean_cor = np.mean(np.array(cor_value))
    return mean_cor, elm

#真假样本混合打乱
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

    
ACC=[]
#FPR=[]
#TPR=[]
batches = []
for top_c in range(20,21):

    train_pos = np.loadtxt('./save/riasec/real_data_train.txt')
    train_negative=[]
    for i in range(70000):
            row = np.random.randint(0, 5, 48).tolist()
            train_negative.append(row)
    print(type(train_negative))
    train_neg = np.array(train_negative)
    
    test_negative=[]
    for i in range(12800):
        row = np.random.randint(0, 5, 48).tolist()
        test_negative.append(row)
    test_negative_random='test_negative_random.txt'
    with open(test_negative_random,'w') as fout:
        for poem in test_negative:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)
            
        
    #训练集真样本
    pos_mean_cor,pos_elm=cor_compute(-1,top_c,train_pos)#正相关
    neg_mean_cor,neg_elm=cor_compute(1,top_c,train_pos)#负相关
    #训练集假样本
    train_f_pos_corr=[]
    for i in range(len(pos_elm)):
        corr = np.corrcoef(train_neg[:,pos_elm[i][0]],train_neg[:,pos_elm[i][1]])[0][1] 
        train_f_pos_corr.append(corr)
    train_f_pos_corr_array = np.array(train_f_pos_corr) 
    train_f_pos_corr_mean = np.mean(train_f_pos_corr_array)   
    #print(train_f_pos_corr_mean)
    '''
    pos_mean_cor2,pos_elm2=cor_compute(-1,top_c,train_neg)#正相关
    neg_mean_cor2,neg_elm2=cor_compute(1,top_c,train_neg)#负相关
    '''
    for total_batch in range(0, 10001, 50):
        acc=0
        #准备测试集
    
        test_fp= './save/riasec/real_data_test.txt'
        '''
        test_positive=test_fp.readlines()
        test_positive_sampling = sample(test_positive,100) #sample
        test_positive_sam='test_sam/D_positive_eval_sample100_'+str(total_batch)+'.txt'
        with open(test_positive_sam,'w') as fout:
            for poem in test_positive_sampling:
                fout.write(poem)
        '''
    
        test_fn='./save/riasec/samples/samples_' + str(total_batch)  + '.txt'
        #'result_J/G_negative_eval_1000_'+str(total_batch)+'.txt'
        #'test_negative_random.txt'
        #
        '''
        test_negative=test_fn.readlines()
        test_negative_sampling = sample(test_negative,100) #sample  
        test_negative_sam='test_sam/D_negative_eval_sample100_'+str(total_batch)+'.txt'
        with open(test_negative_sam,'w') as fout:
            for poem in test_negative_sampling:
                fout.write(poem)
        '''
        test_sentences, test_labels =  load_test_data(test_fp, test_fn)#打乱测试集
        #test_sentences, test_labels =  load_train_pos(test_fn)        
        cont=0
        #fp=0
        #tp=0
        y_labels=[]
        
    
        for test_index, test_elm in enumerate(test_sentences):
            pos_cor_1 = []
            pos_cor_2 = []
            neg_cor_1 = []
            neg_cor_2 = []
            #print(test_elm)
            #test_elm_array = np.array(test_elm)
            for i in range(top_c):
                pos_cor_1.append(test_elm[pos_elm[i][0]])
                pos_cor_2.append(test_elm[pos_elm[i][1]])
                neg_cor_1.append(test_elm[neg_elm[i][0]])
                neg_cor_2.append(test_elm[neg_elm[i][1]])
            pos_cor_1_arr=np.array(pos_cor_1)
            pos_cor_2_arr=np.array(pos_cor_2)
            neg_cor_1_arr=np.array(neg_cor_1)
            neg_cor_2_arr=np.array(neg_cor_2)
            pos_cor_1_pd=pd.Series(pos_cor_1)
            pos_cor_2_pd=pd.Series(pos_cor_2)
            neg_cor_1_pd=pd.Series(neg_cor_1)
            neg_cor_2_pd=pd.Series(neg_cor_2)
            
            pos_corr = np.corrcoef(pos_cor_1_arr,pos_cor_2_arr)
            neg_corr = np.corrcoef(neg_cor_1_arr,neg_cor_2_arr)
           # print(pos_corr[0][1])
    #
            if  abs(pos_corr[0][1]-pos_mean_cor)<=abs(pos_corr[0][1]-train_f_pos_corr_mean): #and neg_corr<=neg_mean_cor:#
                y_labels.insert(test_index,[0,1])#也可以append顺序插入
                
            else:
                y_labels.insert(test_index,[1,0])
            if y_labels[test_index][0] == test_labels[test_index][0]:
                    cont +=1
                    
        acc = cont/len(test_labels)
        ACC.append(acc)
        batches.append(total_batch)
        print( acc, ' ', total_batch)
        #acc = metrics.r2_score(labels, y_labels,multioutput='raw_values')
       # print('acc:',acc)

with open('./save/riasec/acc_result/paps_acc.txt', 'w') as f:
    for i in range(len(ACC)):
        f.write(str(ACC[i]) + ' ' + str(batches[i]) + '\n')
    
    
print(ACC)
