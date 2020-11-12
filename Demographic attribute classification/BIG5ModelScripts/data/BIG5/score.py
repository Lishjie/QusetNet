# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

answers = np.load("answers.npy")
compliance_data = np.load("compliance_data.npy")
noise_data = np.load("noise_data.npy")
features = np.load("features.npy")

        
def BIG5_chioce(answers):
    score = []
    for answer in answers:
        relu = []
        
        relu.append(abs(answer[1]-answer[5])<=1 and answer[1] != 3 and answer[5] != 3)
        relu.append(abs(answer[19]-answer[15])<=1 and answer[19] != 3 and answer[15] != 3)
        relu.append(abs(answer[16]-answer[17])<=1 and answer[16] != 3 and answer[17] != 3)
        
        # assure two answer has same poly, when they has tight connected
        relu.append(abs(answer[44]-answer[49])<=2 \
                    and (answer[44] <=3 and answer[49] <= 3 or answer[44] >= 3 and answer[49] >= 3))
        relu.append(abs(answer[23]-answer[25])<=2 \
                    and (answer[23] <=3 and answer[25] <= 3 or answer[23] >= 3 and answer[25] >= 3))
        relu.append(abs(answer[20]-answer[24])<=2 \
                    and (answer[20] <=3 and answer[24] <= 3 or answer[20] >= 3 and answer[24] >= 3))
        relu.append(abs(answer[24]-answer[26])<=2 \
                    and (answer[24] <=3 and answer[26] <= 3 or answer[24] >= 3 and answer[26] >= 3))
        relu.append(abs(answer[20]-answer[26])<=2 \
                    and (answer[20] <=3 and answer[26] <= 3 or answer[20] >= 3 and answer[26] >= 3))
        
        relu.append(abs(answer[42]-answer[45])>=2 and answer[42] != 3 and answer[45] != 3)
        relu.append(abs(answer[21]-answer[26])>=2 and answer[21] != 3 and answer[26] != 3)
        relu.append(abs(answer[7]-answer[8])>=2 and answer[7] != 3 and answer[8] != 3)
        relu.append(abs(answer[6]-answer[9])>=2 and answer[6] != 3 and answer[9] != 3)
        relu.append(abs(answer[10]-answer[11])>=2 and answer[10] != 3 and answer[11] != 3)
        relu.append(abs(answer[13]-answer[15])>=2 and answer[13] != 3 and answer[15] != 3)
        relu.append(abs(answer[13]-answer[19])>=2 and answer[13] != 3 and answer[19] != 3)
        
        score.append(relu.count(True))
    return score

score = BIG5_chioce(answers)
score_compliance_data = BIG5_chioce(compliance_data)

np.save("score.npy",score)
np.save("score_compliance_data.npy", score_compliance_data)

t = 0
for i in score :
    if i<=10:
        t+=1