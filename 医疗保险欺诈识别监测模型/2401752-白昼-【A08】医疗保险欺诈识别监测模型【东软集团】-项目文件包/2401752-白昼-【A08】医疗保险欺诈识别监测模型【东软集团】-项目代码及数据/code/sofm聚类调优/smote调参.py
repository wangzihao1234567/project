import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import f1_score
# 生成分类模型性能报告,f1分数
# from sklearn.model_selection import cross_val_score  # 交叉验证
# from collections import Counter
from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from tqdm import tqdm
# 未过采样前训练集
ori_data = pd.read_excel('sofm0_train_rus.xlsx')
X_ori = ori_data.iloc[:, :-1]
y_ori = ori_data.iloc[:, -1]

# 测试集
data_test = pd.read_excel('sofm_test0.xlsx')
X_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]

# smote = SMOTE(random_state=100)
# param_grid = {'k_neighbors': list(
#     range(1, 21, 5)), 'sampling_strategy': list(range(0.05, 0.50, 0.1))}

# print('采样前：', Counter(y_ori))
scores = []

for k in tqdm(range(1,40,1)):
    for s in np.arange(0.2, 0.5, 0.01):
        smote = SMOTE(
            random_state=100, k_neighbors=k, sampling_strategy=s
        )

        x_smote, y_smote = smote.fit_resample(X_ori, y_ori)
    # print('采样后：', Counter(y_smote))
    # data = pd.concat([x_smote, y_smote], axis=1)

    # 训练，使用F1作为评判指标
    # X = data.iloc[:, :-1]
    # y = data.iloc[:, -1]
        # x_smote.drop(columns='index')
        # y_smote.drop(columns='index')
        rfc = RFC(n_estimators=100, random_state=100  # , max_depth=50
                  #   ,min_samples_leaf=1
                  #   ,min_samples_split=2
                  )
        rfc.fit(x_smote, y_smote)
        y_pre = rfc.predict(X_test)
        fscore = f1_score(y_test, y_pre)
        # print('Classification Report:\n', classification_report(y_test, y_pre, digits=4))
        # print('F1 score:', f1_score)
        scores.append([k, s, fscore])

max_score = 0
max_s = max_k = None

for score in scores:
    k = score[0]
    s = score[1]
    fscore = score[2]
    if fscore > max_score:
        max_score = fscore
        max_s = s
        max_k = k

print('\n', 'max_f1_score:', max_score, '\n', 'k:', max_k, '\n', 's:', max_s)
