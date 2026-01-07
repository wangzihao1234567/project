import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
import time
from alive_progress import alive_bar
from tqdm import trange
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


#数据集， X是特征矩阵，Y是目标变量（标签）
data = pd.read_excel(r"D:\服务创业\新代码\sofm_3.xlsx")
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

data_test = pd.read_excel(r"D:\服务创业\新代码\sofm_test3.xlsx")
X_test = data_test.iloc[:,:-1]
Y_test = data_test.iloc[:,-1]
def fitness(n_estimators,max_depth, min_samples_leaf, min_samples_split):
    rfc = RandomForestClassifier(n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            min_samples_leaf=int(min_samples_leaf),
                            min_samples_split=int(min_samples_split),
                            random_state=100,
                            n_jobs=-1,
                            #oob_score=True
                            )
    rfc.fit(X, Y)
    y_p_r = rfc.predict(X_test)
    F1_score = f1_score(Y_test,y_p_r)
    return F1_score
up=[122,12,1,2]
down=[138,36,10,10]
min_leaf=5
min_left=5
best_n=0
best_depth=0
best_f1=0
with alive_bar((down[0]-up[0])*(down[1]-up[1])*(down[2]-up[2])*(down[3]-up[3]),force_tty=True) as bar:
    for i in range(up[0],down[0]):
        for j in range(up[1], down[1]):
            for m in range(up[2], down[2]):
                for n in range(up[3], down[3]):
                    f1 = fitness(i, j, m, n)
                    if (f1 > best_f1):
                        time.sleep(0.5)
                        best_n = i
                        best_depth = j
                        min_leaf=m
                        min_left=n
                        best_f1 = f1
                    int_global_best_position = [best_n, best_depth, min_leaf, min_left]
                    time.sleep(0.1)
                    bar()
        print(f"当前最高f1：{best_f1}")
        print(f'当前参数: {int_global_best_position}')
print(f"当前最高f1：{best_f1}")
print(f'当前参数: {int_global_best_position}')



