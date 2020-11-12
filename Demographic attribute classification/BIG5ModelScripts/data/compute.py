import numpy as np
from collections import Counter
test=np.load('G:\\Files\\paper\\r\\classify\\likertelmo-master\\BIG5ModelScripts\\data\\BIG5\\compliance_feature.npy')
test1=[[row[i] for row in test] for i in range(len(test[0]))]
list1=[]
a=len(set(test1[0]))
b=len(set(test1[1]))
c=len(set(test1[2]))
d=len(set(test1[3]))
e=len(set(test1[4]))
f=len(set(test1[5]))
g=len(set(test1[6]))
list1.append(a)
list1.append(b)
list1.append(c)
list1.append(d)
list1.append(e)
list1.append(f)
list1.append(g)
print(list1)
