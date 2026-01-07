from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
df=pd.read_excel('data_train.xlsx')
columns_x = df.columns
df=df.to_numpy()
df_test=pd.read_excel('data_test.xlsx')
df_test=df_test.to_numpy()

som_shape = (2,2)
last_cluster_sofm=som_shape[0]*som_shape[1]
som = MiniSom(som_shape[0], som_shape[1], df.shape[1], sigma=1.4, learning_rate=0.515,
                      neighborhood_function='gaussian', random_seed=10)
som.pca_weights_init(df)
#som.random_weights_init(df)
som.train_random(df, 300, verbose=False)
winner_coordinates = np.array([som.winner(x) for x in df]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

print(cluster_index)
print((pd.Series(cluster_index)).value_counts())
cluster_index=pd.Series(cluster_index)

cluster=pd.DataFrame(cluster_index).value_counts()
cluster=pd.DataFrame(cluster).index.tolist()
cluster=pd.DataFrame(cluster)


heatmap = som.distance_map()  # 生成U-Matrix
fig, ax = plt.subplots()
ax.imshow(heatmap, cmap='bone_r')  # miniSom案例中用的pcolor函数,需要调整坐标
ax.xaxis.set_visible(False)
plt.show()
x_sofm=pd.concat([pd.DataFrame(df,columns=columns_x),pd.DataFrame(cluster_index,columns=['类别'])],axis=1)

winner_coordinates = np.array([som.winner(x) for x in df_test]).T
cluster_index_test = np.ravel_multi_index(winner_coordinates, som_shape)
print((pd.Series(cluster_index_test)).value_counts())
x_sofm_test=pd.concat([pd.DataFrame(df_test,columns=columns_x),pd.DataFrame(cluster_index_test,columns=['类别'])],axis=1)
import os
# 设置文件名
filename = 'D:\服务创业\代码\sofm_train.xlsx'

# 检查文件是否存在
if os.path.exists(filename):
    # 如果文件存在，则删除它
    os.remove(filename)
x_sofm.to_excel('sofm_train.xlsx', index=False)

filename1 = 'D:\服务创业\代码\sofm_test.xlsx'

if os.path.exists(filename1):
    # 如果文件存在，则删除它
    os.remove(filename1)
x_sofm_test.to_excel('sofm_test.xlsx', index=False)