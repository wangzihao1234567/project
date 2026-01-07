import os.path

from imblearn.over_sampling import SMOTE
import pandas as pd
df=pd.read_excel('sofm_train.xlsx')
'''
n,m=df.shape
cluster=pd.DataFrame(df.iloc[:,m-1]).value_counts()
print(cluster)
cluster=pd.DataFrame(cluster).index.tolist()
cluster_list=[]
for i in range(2):
    cluster_list.append(cluster[i][0])
print(cluster_list)

for i in cluster_list:
    filename = f'D:\服务创业\SofmSmote{i}.xlsx'
    if os.path.exists(filename):
        os.remove(filename)
'''
file_name = f'sofm_{0}.xlsx'
group = df.groupby(df.iloc[:,  - 1]).get_group(0).iloc[:, :-1]
x=group.iloc[:,0:-1]
y=group.iloc[:,-1]
print(y.value_counts())
smote = SMOTE(random_state=100
                        , k_neighbors=3
                        ,sampling_strategy=0.31
                        #cluster_balance_threshold=0.05
                        )
x_smote, y_smote = smote.fit_resample(x, y)
df_smote = pd.concat([x_smote, y_smote], axis=1)
df_smote.to_excel(file_name,index=False)

file_name = f'sofm_{1}.xlsx'
group = df.groupby(df.iloc[:,  - 1]).get_group(1).iloc[:, :-1]
x=group.iloc[:,0:-1]
y=group.iloc[:,-1]
print(y.value_counts())
smote = SMOTE(random_state=100
                        , k_neighbors=14
                        ,sampling_strategy=0.34
                        #cluster_balance_threshold=0.05
                        )
x_smote, y_smote = smote.fit_resample(x, y)
df_smote = pd.concat([x_smote, y_smote], axis=1)
df_smote.to_excel(file_name,index=False)

file_name = f'sofm_{2}.xlsx'
group = df.groupby(df.iloc[:,  - 1]).get_group(2).iloc[:, :-1]
x=group.iloc[:,0:-1]
y=group.iloc[:,-1]
print(y.value_counts())
smote = SMOTE(random_state=100
                        , k_neighbors=12
                        ,sampling_strategy=0.17
                        #cluster_balance_threshold=0.05
                        )
x_smote, y_smote = smote.fit_resample(x, y)
df_smote = pd.concat([x_smote, y_smote], axis=1)
df_smote.to_excel(file_name,index=False)


file_name = f'sofm_{3}.xlsx'
group = df.groupby(df.iloc[:,  - 1]).get_group(3).iloc[:, :-1]
x=group.iloc[:,0:-1]
y=group.iloc[:,-1]
print(y.value_counts())
smote = SMOTE(random_state=100
                        , k_neighbors=29
                        ,sampling_strategy=0.25
                        #cluster_balance_threshold=0.05
                        )
x_smote, y_smote = smote.fit_resample(x, y)
df_smote = pd.concat([x_smote, y_smote], axis=1)
df_smote.to_excel(file_name,index=False)