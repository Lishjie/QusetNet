# -*- coding: utf-8 -*-
'''
# @project : WGAN
# @Time    : 2019/12/10 20:34
# @Author  : plzhao
# @FileName: fig.py
# @Software: PyCharm
'''

import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import interpolate

# 路径配置
ma_path = input("ma path:")
d_acc_path = input("d acc path:")
pa_acc_path = input("pa acc path:")
ps_acc_path = input("ps acc path:")
odd_acc_path = input("odd acc path")
# longest_acc_path = input("longest acc path:")
rc_acc_path = input("rc acc path:")
rr_acc_path = input("rr acc path:")
per_acc_path = input("per acc path:")

# params config
plt.subplots(figsize=(11, 5))
ax = plt.gca()
box = ax.get_position()
plt.xlim(xmax=7000)
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}

# acc_pattern = re.compile('acc: (0.\d+)')
# with open('nohup.out', 'r') as infile:
#     ma_out = infile.read()
# ma_result = acc_pattern.findall(ma_out)
# ma_result = [float(res) for res in ma_result]
ma_result = np.loadtxt(ma_path)
# D_acc = np.loadtxt('./save/riasec/acc_result/D_acc_8200.txt')
# D_acc = np.loadtxt('./save/riasec/acc_result/D_acc_8200.txt')
D_acc = np.loadtxt(d_acc_path)
# ma_acc = np.loadtxt('./save/riasec/acc_result/ma_acc.txt')
# pa_acc = np.loadtxt('./save/riasec/acc_result/pa_acc.txt')
pa_acc = np.loadtxt(pa_acc_path)
# paps_acc = np.loadtxt('./save/riasec/acc_result/paps_acc.txt')
ps_acc = np.loadtxt(ps_acc_path)
# odd_acc = np.loadtxt('./save/riasec/acc_result/odd_acc.txt')
odd_acc = np.loadtxt(odd_acc_path)
# longest_acc = np.loadtxt('./save/riasec/acc_result/longest_acc.txt')
# longest_acc = np.loadtxt(longest_acc_path)
# rc_acc = np.loadtxt('./save/riasec/acc_result/rc_acc.txt')
rc_acc = np.loadtxt(rc_acc_path)
# rr_acc = np.loadtxt('./save/riasec/acc_result/rr_acc.txt')
rr_acc = np.loadtxt(rr_acc_path)
# per_acc = np.loadtxt('./save/riasec/acc_result/per_acc.txt')
per_acc = np.loadtxt(per_acc_path)

def data_smooth(x, y):
    # pinghua quxian chuli 
    # tck = interpolate.splrep(x, y, k = 3, s = 1)
    # y = interpolate.splev(x, tck, der = 0)

    x_list = []
    y_list = []
    x_list.append(x[0])
    y_list.append(y[0])
    # qu pingjun
    for index in range(0, len(y)-1, 3):
        x_temp = 0
        y_temp = 0
        num = 0
        for length in range(3):
            if index + length + 2 >= len(y):
                break
            else:
                x_temp += x[index + length]
                y_temp += y[index + length]
                num += 1
        if num != 0:
            x_list.append(x_temp / num)
            y_list.append(y_temp / num)
        else:
            continue
    x_list.append(x[-1])
    y_list.append(y[-1])

    # return x, y
    return np.array(x_list), np.array(y_list)

p1_X, p1_y = data_smooth(D_acc[:160, 1], D_acc[:160, 0])
p1, = plt.plot(p1_X, p1_y)
# p1, = plt.plot(D_acc[0:40,1],D_acc[0:40,0])
# p1, = plt.plot(D_acc[0:,1],D_acc[0:,0])
# p2, = plt.plot(ma_acc[:, 1], ma_acc[:, 0])
p2_X, p2_y = data_smooth(ma_result[:160, 1], ma_result[:160, 0])
p2, = plt.plot(p2_X, p2_y)
p3_X, p3_y = data_smooth(pa_acc[:160, 1], pa_acc[:160, 0])
p3, = plt.plot(p3_X, p3_y)
p4_X, p4_y = data_smooth(ps_acc[:160, 1], ps_acc[:160, 0])
p4, = plt.plot(p4_X, p4_y)
p5_X, p5_y = data_smooth(odd_acc[:160, 1], odd_acc[:160, 0])
p5, = plt.plot(p5_X, p5_y)
# p6_X, p6_y = data_smooth(longest_acc[:160, 1], longest_acc[:160, 0])
# p6, = plt.plot(p6_X, p6_y)
p7_X, p7_y = data_smooth(rc_acc[:160, 1], rc_acc[:160, 0])
p7, = plt.plot(p7_X, p7_y)
# p8_X, p8_y = data_smooth(rr_acc[:160, 1], rr_acc[:160, 0])
# p8, = plt.plot(p8_X, p8_y)
p9_X, p9_y = data_smooth(per_acc[:160, 1], per_acc[:160, 0])
p9, = plt.plot(p9_X, p9_y)

plt.tick_params(labelsize=16)
# plt.legend([p1, p2, p3, p4, p5, p6, p7, p8, p9],
#            ["QuesD", "MD", "PA", "PAPS", "ODD", "LongesT", "RC", "RR", "PER"],
#            loc='best', prop=font1)
plt.legend([p1, p2, p3, p4, p5, p7, p9],
           ["QuesD", "MD", "PA", "PS", "ODD", "RC", "RR", "PER"],
           prop=font1, bbox_to_anchor=(1.20, 1), loc='upper right', borderaxespad = 0.)
# plt.figure(figsize=(8, 10.5))
plt.savefig('riasec_result9000.png', dpi=600)
plt.show()

# for i in FILES:
#    plt.scatter(D_real,Y ,c = 'r',marker = 'o')
#    plt.scatter(i,Y ,c = 'b',marker = 'o')
#    plt.xlim((-3, 2))
#    plt.xticks([])  # ignore xticks
#    plt.ylim((0, 10))
#    plt.yticks([])  # ignore yticks
#    plt.show()
