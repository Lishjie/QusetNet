# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import toexce
import matplotlib.pyplot as plt
import seaborn as sns

computer = toexce.computer
# BIG5
path = "data.csv"
ans_csv = pd.read_csv(path, sep="\t")
data = ans_csv.copy()


#data=data[~data['VCL6'].isin([1])]
#data=data[~data['VCL9'].isin([1])]
#data=data[~data['VCL12'].isin([1])]
#
#for col in data.iloc[:, :48].columns:
#    data=data[~data[col].isin([0])]
#    
#for col in data.iloc[:, 51:61].columns:
#    data=data[~data[col].isin([0])]
#
#for col in data.iloc[:, 77:87].columns:
#    data=data[~data[col].isin([0])]

answers = data.iloc[:, :48]

#feature_1 = ans_csv.iloc[:, 48:51]
feature_2 = data.iloc[:, 51:61]
feature_3 = data.iloc[:, 61:77]
feature_4 = data.iloc[:, 77:-2]
#feature_2.index = range(0,len(feature_2))
#feature_4.index = range(0,len(feature_4))

#########################################
ages = list(feature_4["age"].copy())
for i, age in enumerate(ages):
    if age < 18:
        ages[i] = 0
    elif age < 28 and age >= 18:
        ages[i] = 1
    elif age < 40 and age >= 28:
        ages[i] = 2
    elif age < 66 and age >= 40:
        ages[i] = 3
    else:
        ages[i] = 4

feature_4["age"] = pd.Series(ages)

#########################################
country = list(feature_4["country"].copy())
country_list = set(country)

dic = {}
dic['OO'] = 0
for cou in country_list:
    dic[cou] = country.count(cou)

for cou in country_list:
    if cou not in computer:
        dic['OO'] += dic.pop(cou)

cou_dic = {}
for i, cd in enumerate(list(dic.keys())):
    cou_dic[cd] = i
for i, age in enumerate(country):
    try:
        country[i] = cou_dic[age]
    except BaseException:
        country[i] = cou_dic["OO"]
        
feature_4["country"] = pd.Series(country)
#########################################
uniqueNetworkLocation = list(feature_4["uniqueNetworkLocation"].copy())
uniqueNetworkLocation = [x-1 for x in uniqueNetworkLocation]
feature_4["uniqueNetworkLocation"] = pd.Series(uniqueNetworkLocation)
#########################################

familysize = list(feature_4["familysize"].copy())
#familysize = [(lambda x: x if x< 10 else 10)(x) for x in familysize]
new_familysize = []
for num in familysize:
    if int(num)>9:
        new_familysize.append(10)
    else:
        new_familysize.append(num)
feature_4["familysize"] = pd.Series(new_familysize)
#########################################

for i,col in enumerate(feature_4.iloc[:, :].columns):
    print(col)
    if col in ["age","country","uniqueNetworkLocation","familysize","source","married"]:
        feature_4[col] =feature_4[col]+1


feature = pd.concat([feature_2,feature_4],axis=1)

#np.save("answers.npy",answers)
#np.save("features.npy",feature)
#answers.to_excel("answers.xlsx", index=None)
#feature.to_excel("features.xlsx", index=None)



max_list = []
for i in range(25):
    max_list.append(feature.iloc[:,i].max())
print(max_list)

#for i,col in enumerate(feature_4.iloc[:, :].columns):
#    print(col,list(feature[col]).count(0),list(feature[col]).count(1))

# 开始计算序变量之间的关联度
problem_mse_list = []
problem_mse_array = np.full((48, 48), -1.0,)
mse_less_2 = {}
mse_less_1point5 = {}
mse_more_7 = {}
mse_more_8 = {}

answers_np = answers.to_numpy()
for i in range(answers_np.shape[1] - 1):
    for j in range(i+1, answers_np.shape[1]):
        mse = np.sqrt(np.mean(np.square(answers_np[:, i] - answers_np[:, j])))
        problem_mse_list.append(mse)
        problem_mse_array[i][j] = mse
        # 单独记录强相关问题和强相反问题
        if mse < 2:
            mse_less_2[str(i) + '' + str(j)] = mse
            if mse < 1.5:
                mse_less_1point5[str(i) + '' + str(j)] = mse
        elif mse > 7:
            mse_more_7[str(i) + '' + str(j)] = mse
            if mse > 8:
                mse_more_8[str(i) + '' + str(j)] = mse

# 根据取值区间的问卷个数来估计强相关区间、强相反区间的范围
length = 0  # 关联问题的总个数
len_max = 0 # 强关联问题的个数
len_min = 0 # 强相反问题的个数
for i in range(answers_np.shape[1] - 1):
    data = np.array(problem_mse_array[i, i+1:48], dtype=np.float32)
    length += data.shape[0]
    len_max += np.array(np.where((data[:]>2.84) & (data[:]<=4))).shape[1] # 强相反问题的个数
    len_min += np.array(np.where((data[:]>=0) & (data[:]<1.00))).shape[1] # 强相关问题的个数

print(length)
print(len_max)
print(len_max / length)
print(len_min)
print(len_min / length)
# 使用图表展示问题之间的关联矩阵
#f, (ax1) = plt.subplots(figsize=(50,50),nrows=1)
#sns.heatmap(problem_mse_array, annot=True, ax=ax1, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
