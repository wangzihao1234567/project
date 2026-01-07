from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from tqdm import tqdm
# 未过采样前训练集
ori_data = pd.read_excel('../data/data_train.xlsx')
X_ori = ori_data.iloc[:, :-1]
y_ori = ori_data.iloc[:, -1]

# 测试集
data_test = pd.read_excel('../data/data_test.xlsx')
X_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]


scores = []

for k in tqdm(range(5, 20)):
    for s in np.arange(0.10, 0.21, 0.01):
        smote = SMOTE(
            random_state=100, k_neighbors=k, sampling_strategy=s
        )

        x_smote, y_smote = smote.fit_resample(X_ori, y_ori)

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
