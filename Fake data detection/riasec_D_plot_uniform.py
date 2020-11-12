import numpy as np
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
'''
# @project : WGAN
# @Time    : 2019/12/10 20:34
# @Author  : plzhao
# @FileName: fig.py
# @Software: PyCharm
'''


'''
which_D = 50
FILES = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,150,200,250,300,350,400,450,500,600,700,800,900,1000,1500,2000,4000,6000]

D_real = np.loadtxt('./save/big5/D_out'+str(which_D)+'/real.txt')
for file in FILES:
    D_fake = np.loadtxt('./save/big5/D_out'+str(which_D)+'/'+str(file)+'.txt')
    threshold = (np.sum(D_real)+np.sum(D_fake))/(len(D_real)+len(D_fake))
    right_num = 0
    for value in D_real:
        if value>=threshold:
            right_num+=1
    for value in D_fake:
        if value<=threshold:
            right_num+=1
    acc = right_num/(len(D_real)+len(D_fake))

    print(acc,'            ',file)
    
'''

which_D = 9000
# f1 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150,
#       200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1500, 2000]
# f2 = [i for i in range(2500, 8000, 100)]
#f3 = [i for i in range(10000,15001,1000)]
f1 = [i for i in range(0, 10001, 50)] + [i for i in range(10000, 50000, 1000)]
# FILES = f1+f2
FILES = f1
#
ACC = []
batches = []
# for file in FILES:
#
#    acc_D = []
#    for i  in range(10):
#        this_D = which_D+100*i +100
#        D_real = np.loadtxt('./save/big5/model_Dout_1215/D_out'+str(this_D)+'/real.txt')
#        D_fake = np.loadtxt('./save/big5/model_Dout_1215/D_out'+str(this_D)+'/'+str(file)+'.txt')
#        threshold = (np.sum(D_real)+np.sum(D_fake))/(len(D_real)+len(D_fake))
#        right_num = 0
#        for value in D_real:
#            if value>=threshold:
#                right_num+=1
#        for value in D_fake:
#            if value<=threshold:
#                right_num+=1
#        acc_each_D = right_num/(len(D_real)+len(D_fake))
#        acc_D.append(acc_each_D)
#    acc = sum(acc_D)/len(acc_D)
#    print(acc,' ',file)
#
#    ACC.append(acc)
#    batches.append(file)
# with open('./save/big5/acc_result/D_acc_1215.txt', 'w') as f:
#    for i in range(len(ACC)):
#        f.write(str(ACC[i]) + ' ' + str(batches[i]) + '\n')

# 路径配置
real_data = input("real data:")
fake_data = input("fake data:")
save_path = input("save path:")

for file in range(1, 202):
    # D_real = np.loadtxt(
    #     './save/riasec/model_Dout/D_out'+str(which_D)+'/real.txt')
    D_real = np.loadtxt(real_data)
    # D_fake = np.loadtxt('./save/riasec/model_Dout/D_out' +
    #                     str(which_D)+'/'+str(file)+'.txt')
    D_fake = np.loadtxt(fake_data+str(file)+' .txt')
    threshold = (np.sum(D_real)+np.sum(D_fake))/(len(D_real)+len(D_fake))
    right_num = 0
    for value in D_real:
        if value >= threshold:
            right_num += 1
    for value in D_fake:
        if value <= threshold:
            right_num += 1
    acc = right_num/(len(D_real)+len(D_fake))

    print(acc, ' ', file)
    ACC.append(acc)
    batches.append(file)
# with open('./save/riasec/acc_result/D_acc_8200.txt', 'w') as f:
with open(save_path, 'w') as f:
    for i in range(len(ACC)):
        f.write(str(ACC[i]) + ' ' + str(f1[i]) + '\n')
