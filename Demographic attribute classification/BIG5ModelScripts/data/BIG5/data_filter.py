# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:48:18 2019

@author: lishijie
@description: 用于筛选噪声数据，留下合规数据
"""
import numpy as np
#import Config
import copy
import matplotlib.pyplot as plt
import seaborn as sns

# 获取数据源配置信息
#config = Config.Big5Config()
answers = np.load(
    "/home/nlp/Desktop/likertelmo/BIG5ModelScripts/data/BIG5/answers.npy")    # 所有问卷的信息
features = np.load(
    "/home/nlp/Desktop/likertelmo/BIG5ModelScripts/data/BIG5/features.npy")  # all features
# 所有问卷的得分(得分<=10的为噪声数据)
score = np.load(
    "/home/nlp/Desktop/likertelmo/BIG5ModelScripts/data/BIG5/score.npy")
classes = [1 if x > 10 else 0 for x in list(score)]  # 每条数据的标签(是否噪声)

# 开始将噪声数据和合规数据分别存入不同的数组中
compliance_data = []
compliance_feature = []
noise_data = []
noise_feature = []
problem_mse_dict = {}
problem_mse_array = np.full((50, 50), -1.0,)
problem_mse_list = []
mse_less_2 = {}
mse_less_1point5 = {}
mse_more_3 = {}
mse_more_3point5 = {}


def answer_map(answers):
    for i in range(answers.shape[0]):
        for j in range(answers.shape[1]):
            if answers[i][j] == 1 or answers[i][j] == 2:
                answers[i][j] = 1
            elif answers[i][j] == 3:
                answers[i][j] = 2
            elif answers[i][j] == 4 or answers[i][j] == 5:
                answers[i][j] = 3
    return answers

for rowNum in range(answers.shape[0]):
    if classes[rowNum] == 1:
        compliance_data.append(answers[rowNum][:])
        compliance_feature.append(features[rowNum][:])
    else:
        noise_data.append(answers[rowNum][:])
        noise_feature.append(features[rowNum][:])
        
compliance_data_ = np.array(compliance_data, dtype=np.int)
#np.save("compliance_data.npy", compliance_data)
# 每次将生成的数据随即打乱
compliance_data_copy = copy.copy(compliance_data)
np.random.shuffle(compliance_data_copy)
#np.save("compliance_data.npy", compliance_data)
noise_data_ = np.array(noise_data, dtype=np.int)
#np.save("noise_data.npy", noise_data)
#np.save("compliance_feature.npy", compliance_feature)
#np.save("noise_feature.npy", noise_feature)

# 将答案的类型映射为3类
#answer_map(compliance_data_)

# 计算50个性格测试问题的关联性，使用均方误差最为评价标准
compliance_data__ = compliance_data_[0:6468]
for i in range(compliance_data__.shape[1]-1):
    for j in range(i+1, compliance_data__.shape[1]):
        # 标准差
#        mse = np.std(compliance_data__[:, i] - compliance_data__[:, j], ddof=1)
        # 均方误差
        mse = np.sqrt(np.mean(np.square(compliance_data__[:, i] - compliance_data__[:, j])))
        problem_mse_list.append(mse)
        problem_mse_dict[str(i) + ' ' + str(j)] = mse
        problem_mse_array[i][j] = mse
        # 用于记录强相关、强相反问题
        if mse < 2:
            mse_less_2[str(i) + ' ' + str(j)] = mse
            if mse < 1.5:
                mse_less_1point5[str(i) + ' ' + str(j)] = mse
        elif mse > 3:
            mse_more_3[str(i) + ' ' + str(j)] = mse
            if mse > 3.5:
                mse_more_3point5[str(i) + ' ' + str(j)] = mse
                
length = 0
len_max = 0
len_min = 0
for i in range(compliance_data_.shape[1] - 1):
    data = np.array(problem_mse_array[i, i+1:50], dtype=np.float32)
    length += data.shape[0]
    len_max += np.array(np.where((data[:]>2.84) & (data[:]<=4))).shape[1] # 强相关问题的个数
    len_min += np.array(np.where((data[:]>=0) & (data[:]<1.10))).shape[1] # 强相反问题的个数
print(length)
print(len_max)
print(len_max / length)
print(len_min)
print(len_min / length)

#pros = np.array(problem_mse_list)
#plt.figure(figsize=(10, 12))
#plt.hist(pros,
#         bins=500,
#         normed=1,
#         facecolor="red",
#         edgecolor="red",
#         alpha=0.7)
#plt.savefig('problem_mse.png')
#plt.show()
# 根据规则取出问题关联矩阵中的强相关问题和强相反问题
#strong_relation = np.full((50, 50), -1.0,)
#for row, row_val in enumerate(problem_mse_array):
#    for col, col_val in enumerate(row_val):
#        if col_val>=0 and col_val<1.00 or col_val>2.84 and col_val<=4:
#            strong_relation[row][col] = col_val
##            
#f, (ax1) = plt.subplots(figsize=(50,50),nrows=1)
#sns.heatmap(strong_relation, annot=True, ax=ax1)
#sns.heatmap(problem_mse_array, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
