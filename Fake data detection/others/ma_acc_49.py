# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:32:34 2019

@author: zhuyu
"""

import numpy as np
from sklearn.model_selection import KFold
#from scipy.spatial.distance import pdist
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from random import sample
from sklearn import metrics
from sklearn.metrics import f1_score
from time import strftime, localtime

# 打印当前时间
def printTime():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return
test_sample_num = 640
train_sample_Pnum = 100
train_sample_Nnum = 100
query_num = 48

def int_unk(x):
    try:
        value_int = int(x)
    except:
        value_int = np.random.randint(0, 5)
    return value_int

#真假样本混合打乱
def load_test_data(positive_file, negative_file):
    # Load data
    positive_examples = []
    negative_examples = []
    with open(positive_file)as fin:
        fin_list = []
        for line in fin:
            fin_list.append(line)
        for line in sample(fin_list,test_sample_num):
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            positive_examples.append(parse_line)
    with open(negative_file)as fin:
        fin_list = []
        for line in fin:
            fin_list.append(line)
        for line in sample(fin_list,test_sample_num):
            line = line.strip()
            line = line.split()
            parse_line = [int_unk(x) for x in line]
            if len(parse_line) == query_num:
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
        fin_list = []
        for line in fin:
            fin_list.append(line)
        for line in sample(fin_list,train_sample_Pnum):
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            positive_examples.append(parse_line)
    labels = [[0, 1] for _ in positive_examples]
    sentences = np.array(positive_examples)
    return sentences, labels
   
ACC=[]
#FPR=[]
#TPR=[]
#准备训练集真样本
    
train_fp=np.loadtxt('./save/riasec/real_data_train.txt')
train_positive=train_fp.tolist()
train_positive = sample(train_positive,train_sample_Pnum)
print(train_positive)
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
    
#准备训练集假样本 c
 
batches = []
train_negative=[]
for i in range(train_sample_Nnum):
    row = np.random.randint(0, 5, query_num).tolist()
    train_negative.append(row)
print(type(train_negative))

    #train_sentences, train_labels =  load_train_data(train_fp, train_nclearegative_sam)

for total_batch in range(0,10001,50):
    acc=0
         
#准备测试集'
    test_fp='./save/riasec/real_data_test.txt'
    #'./save/riasec_ma/real_data_test.txt'
    '''
    test_positive=test_fp.readlines()
    test_positive_sampling = sample(test_positive,100) #sample
    test_positive_sam='test_sam/D_positive_eval_sample100_'+str(total_batch)+'.txt'
    with open(test_positive_sam,'w') as fout:
        for poem in test_positive_sampling:
            fout.write(poem)
    '''

    test_fn='./save/riasec/samples/samples_'+str(total_batch)+'.txt'
    # test_fn = './save/samples_50.txt'
    # test_fn = './save/riasec/test_negative_random.txt'
    #'test_negative_random.txt'
    
    #'test_negative_random.txt'
    
    '''
    test_negative=test_fn.readlines()
    test_negative_sampling = sample(test_negative,100) #sample  
    test_negative_sam='test_sam/D_negative_eval_sample100_'+str(total_batch)+'.txt'
    with open(test_negative_sam,'w') as fout:
        for poem in test_negative_sampling:
            fout.write(poem)
    '''
    test_sentences, test_labels =  load_test_data(test_fp, test_fn)#打乱测试集
    #test_sentences, test_labels =  load_train_pos(test_fp)        
    cont=0
    #fp=0
    #tp=0
    y_labels=[]
    
    print('testing strat:~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for test_index, test_elm in enumerate(test_sentences):

        printTime()
        
        distances_pos=[]
        for i in range(len(train_positive)):
            x = np.array(train_positive)
            xt=x.T
            D=np.cov(xt)
            invD=np.linalg.inv(D)
            tp=test_elm-train_positive[i]
            distances_pos.append(np.sqrt(np.dot(np.dot(tp,invD),tp.T)))
        mean_distances_pos = np.mean(distances_pos) 
        print(np.mean(distances_pos))
        distances_neg=[]
        for i in range(len(train_negative)):
            x = np.array(train_negative)
            xt=x.T
            D=np.cov(xt)
            invD=np.linalg.inv(D)
            tp=test_elm-train_negative[i]
            distances_neg.append(np.sqrt(np.dot(np.dot(tp,invD),tp.T)))
        mean_distances_neg = np.mean(distances_neg) 
        print(np.mean(distances_neg))
        if mean_distances_pos<= mean_distances_neg:
            y_labels.insert(test_index,[0,1])#也可以append顺序插入
            
        else:
            y_labels.insert(test_index,[1,0])
        if y_labels[test_index][0] == test_labels[test_index][0]:
                cont +=1
                
    acc = cont/len(test_labels)

   
    #acc = metrics.r2_score(labels, y_labels,multioutput='raw_values')
    print('acc:',acc)
    ACC.append(acc)
    batches.append(total_batch)

with open('./save/riasec_ma/ma_acc.txt', 'w') as f: 
        for i in range(len(ACC)):
            f.write(str(ACC[i])+' '+ str(batches[i]) +'\n')

 
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