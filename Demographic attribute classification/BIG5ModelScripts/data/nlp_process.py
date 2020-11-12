import pandas as pd
import re
#编写此程序发现的自己的问题
#（1）正则表达式还不熟悉
#（2）今天犯了一个低级错误，浪费了很多时间，就是变量名指定的和函数名重复，读了好多遍程序没找到错误。
# data=pd.read_excel('C:\\Users\\Administrator\\Desktop\\data.xlsx',header=0)
# list1=list(data['para_content'])
# print(list1[0])
after_list1 = []
list1=['Experiments were run on three domains to compare learning using the variational approxima    tion with learning using exact inference under the junction tree algorithm. Two activity r    ecognition datasets were used . In these experiments, a robot converted its raw laser read    ings into x-y positions of a person walking. The experiments were conducted in two separat    e domains: a small laboratory/cubicle environment and an entryway inside the University of     Massachusetts Computer Science Department. There were six different trajectories in the l    aboratory environment and eight in the entryway. The length of a typical sequence in these     domains was approximately 20 timesteps. To test the variational technique in a larger dom    ain, a third experiment was conducted using synthetic data. The dataset contained sequence    s of latitude-longitude readings intended to represent airline routes. There were fifteen     different routes in this domain where the longest sequence was approximately 250 timesteps    . A picture of the state space with datapoints for all fifteen routes is shown in Figure 4    .']
for each in list1:
    _after_each = re.findall('[A-Za-z0-9\+\-\(\)\&\ ]', each, re.S)
    _after_each = "".join(_after_each)
    _after_each = ' '.join(_after_each.split())
    after_list1.append(_after_each)
# print(after_list1)

def py_strip(p_str, d_str):
    temp = re.search(r'[^('+d_str+')].*',p_str).group()
    res = re.search(r'.*[^('+d_str+')]',temp).group()
    return res
after2=[]


for each in after_list1:
    #另一种去除首尾字符的方法
    # each.strip("+")
    # each.strip("-")
    # each.strip("&")
    if(each != " "):
        each = py_strip(each, '+')
        each = py_strip(each, '-')
        each = py_strip(each, '&')
        if((each[-1] == '+') or (each[-1] == '-') or (each[-1] == '&')):
            each = py_strip(each, '+')
            each = py_strip(each, '-')
            each = py_strip(each, '&')
        if ((each[-1] == '+') or (each[-1] == '-') or (each[-1] == '&')):
            each = py_strip(each, '+')
            each = py_strip(each, '-')
            each = py_strip(each, '&')
        after2.append(each)


#split and concat
def cuttingText(text,num,textlist=[]):
    # about=[]
    # textlist = []          ## 空列表
    # about=textlist
    text1=text
    if(text1!=[] and len(text1)>=num):
        textlist.append(text[0:num])
        # print(textlist)
        # about.append(textlist)
        # print(about)
        text1=text[1:]
        cuttingText(text1, num,textlist=textlist)
    return textlist
totallist=[]
for each in after_list1:
    # print(each)
    eachlist = each.split(" ")
    # print(eachlist)
    lenth = len(eachlist) #这里一直报错原来是因为变量和函数重名了
    j=1
    while(j<=4):
        textlist = cuttingText(eachlist, j)
        # totallist.append(textlist)
        print(textlist)
        #可以作为句子输出
        # s = ''
        # for k in textlist:
        #     m=''
        #     for every in k:
        #         m=m+" "+every
        #     s=s+","+m
        # print("数量为"+str(j)+"的：" + s)
        j=j+1
        # print(totallist)
