import os.path
#训练集
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
df=pd.read_excel(r'sofm_train.xlsx')
n,m=df.shape
#sum_cluster=max(df.iloc[:,m-1])
cluster=pd.DataFrame(df.iloc[:,m-1]).value_counts()
print(cluster)
cluster=pd.DataFrame(cluster).index.tolist()
cluster_list=[]
for i in range(len(cluster)):
    cluster_list.append(cluster[i][0])
print(cluster_list)

for i in cluster_list:
    file_name_original = f'sofm_train{i}.xlsx'
    group = df.groupby(df.iloc[:, m - 1]).get_group(i).iloc[:, :-1]
    group.to_excel(file_name_original, index=False)

#测试集
df=pd.read_excel(r'sofm_test.xlsx')
n,m=df.shape
#sum_cluster=max(df.iloc[:,m-1])
cluster=pd.DataFrame(df.iloc[:,m-1]).value_counts()
print(cluster)
cluster=pd.DataFrame(cluster).index.tolist()
cluster_list=[]
for i in range(len(cluster)):
    cluster_list.append(cluster[i][0])
print(cluster_list)

for i in cluster_list:
    file_name_original = f'sofm_test{i}.xlsx'
    group = df.groupby(df.iloc[:, m - 1]).get_group(i).iloc[:, :-1]
    group.to_excel(file_name_original, index=False)

