# -*- coding: utf-8 -*-[14, 5, 3, 4, 4, 5, 25]
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd


# BIG5
path = "/home/nlp/Desktop/likertelmo/BIG5ModelScripts/data/BIG5/data.csv"
ans_csv = pd.read_csv(path, sep="\t")
feature = ans_csv.iloc[:, :7].drop(19064)
answers = ans_csv.iloc[:, 7:].drop(19064)


ages = list(feature["age"].copy())
for i, age in enumerate(ages):
    if age < 18:
        ages[i] = 1
    elif age < 28 and age >= 18:
        ages[i] = 2
    elif age < 40 and age >= 28:
        ages[i] = 3
    elif age < 66 and age >= 40:
        ages[i] = 4
    else:
        ages[i] = 5

feature["age"] = pd.Series(ages)

country = list(feature["country"].copy())
country_list = set(country)

dic = {}
dic['OO'] = 0
for cou in country_list:
    dic[cou] = country.count(cou)

dic['OO'] = dic.pop("(nu")
for cou in list(dic.keys()):
    if dic[cou] < 90:
        dic['OO'] += dic.pop(cou)

cou_dic = {}
for i, cd in enumerate(list(dic.keys())):
    cou_dic[cd] = i


for i, age in enumerate(country):
    try:
        country[i] = cou_dic[age]
    except BaseException:
        country[i] = cou_dic["OO"]

feature["country"] = pd.Series(country)

#########feature####################
for col in feature.iloc[:, :].columns:
    print(col)
    if col not in ["age","source"]:
        feature[col] =feature[col]+1

#########answers check####################
max_list_ans = []
for i in range(50):
    max_list_ans.append(answers.iloc[:,i].max())
print(max_list_ans)

min_list_ans = []
for i in range(50):
    min_list_ans.append(answers.iloc[:,i].min())
print(min_list_ans)

#########feature check####################
max_list = []
for i in range(7):
    max_list.append(feature.iloc[:,i].max())
print(max_list)

min_list = []
for i in range(7):
    min_list.append(feature.iloc[:,i].min())
print(min_list)


answers_all = np.array(answers,dtype=np.int32)
feature_all = np.array(feature,dtype=np.int32)  

# np.save("answers.npy",answers_all)
# np.save("features.npy",feature_all)



