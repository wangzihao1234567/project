
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
from sklearn import metrics
data_0 = pd.read_excel('D:\服务创业\新代码\sofm_0.xlsx')
data_1 = pd.read_excel('D:\服务创业\新代码\sofm_1.xlsx')
data_2 = pd.read_excel('D:\服务创业\新代码\sofm_2.xlsx')
data_3 = pd.read_excel('D:\服务创业\新代码\sofm_3.xlsx')
from sklearn.ensemble import RandomForestClassifier as RFC  # 随机森林分类
import warnings
warnings.filterwarnings("ignore")
X_0 = data_0.iloc[:,:-1]
y_0 = data_0.iloc[:,-1]
X_3 = data_3.iloc[:,:-1]
y_3 = data_3.iloc[:,-1]
X_1 = data_1.iloc[:,:-1]
y_1 = data_1.iloc[:,-1]
X_2 = data_2.iloc[:,:-1]
y_2 = data_2.iloc[:,-1]
test_0=pd.read_excel('D:\服务创业\新代码\sofm_test0.xlsx')
test_1=pd.read_excel('D:\服务创业\新代码\sofm_test1.xlsx')
test_2=pd.read_excel('D:\服务创业\新代码\sofm_test2.xlsx')
test_3=pd.read_excel('D:\服务创业\新代码\sofm_test3.xlsx')
x_test_0=test_0.iloc[:,:-1]
y_test_0=test_0.iloc[:,-1]
x_test_2=test_2.iloc[:,:-1]
y_test_2=test_2.iloc[:,-1]
x_test_1=test_1.iloc[:,:-1]
y_test_1=test_1.iloc[:,-1]
x_test_3=test_3.iloc[:,:-1]
y_test_3=test_3.iloc[:,-1]
y_test=pd.concat([y_test_0,y_test_1],axis=0)
y_test=pd.concat([y_test,y_test_2],axis=0)
y_test=pd.concat([y_test,y_test_3],axis=0)
rfc_0 = RFC(n_estimators=22
            , max_depth=11
            , min_samples_leaf=5
            , min_samples_split=5
            , random_state=100
            )
rfc_0.fit(X_0, y_0)

rfc_1 = RFC(n_estimators=139
            , max_depth=30
               ,min_samples_leaf=1
               ,min_samples_split=169
            , random_state=100
            )
rfc_1.fit(X_1, y_1)

rfc_2 = RFC(n_estimators=58
            , max_depth=22
            , min_samples_leaf=1
            , min_samples_split=2
            , random_state=100
            )
rfc_2.fit(X_2, y_2)

rfc_3 = RFC(n_estimators=123
            , max_depth=27
               ,min_samples_leaf=2
               ,min_samples_split=2
            , random_state=100
            )
rfc_3.fit(X_3, y_3)

ypre_0 = rfc_0.predict(x_test_0)
print(f'准确率为{metrics.accuracy_score(y_test_0, ypre_0)}，recall为{metrics.recall_score(y_test_0, ypre_0)}，F1值为{metrics.f1_score(y_test_0, ypre_0)}，precision为{metrics.precision_score(y_test_0, ypre_0)}')
ypre_1 = rfc_1.predict(x_test_1)
print(f'准确率为{metrics.accuracy_score(y_test_1, ypre_1)}，recall为{metrics.recall_score(y_test_1, ypre_1)}，F1值为{metrics.f1_score(y_test_1, ypre_1)},precision为{metrics.precision_score(y_test_1, ypre_1)}')
ypre_2=rfc_2.predict(x_test_2)
print(f'准确率为{metrics.accuracy_score(y_test_2, ypre_2)}，recall为{metrics.recall_score(y_test_2, ypre_2)}，F1值为{metrics.f1_score(y_test_2, ypre_2)},precision为{metrics.precision_score(y_test_2, ypre_2)}')
ypre_3=rfc_3.predict(x_test_3)
print(f'准确率为{metrics.accuracy_score(y_test_3, ypre_3)}，recall为{metrics.recall_score(y_test_3, ypre_3)}，F1值为{metrics.f1_score(y_test_3, ypre_3)},precision为{metrics.precision_score(y_test_3, ypre_3)}')
y_pre=pd.concat([pd.DataFrame(ypre_0),pd.DataFrame(ypre_1)],axis=0)
y_pre=pd.concat([y_pre,pd.DataFrame(ypre_2)],axis=0)
y_pre=pd.concat([y_pre,pd.DataFrame(ypre_3)],axis=0)
f1 = metrics.f1_score(y_test, y_pre)
print(f'准确率为{metrics.accuracy_score(y_test, y_pre)}，recall为{metrics.recall_score(y_test, y_pre)}，F1值为{f1},precision为{metrics.precision_score(y_test, y_pre)}')
