import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
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

#定义fitness函数
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

# PSO参数设置
num = 31  # 粒子数
max_iterations = 30  # 最大迭代次数
fitnessbest = []

# 定义优化问题的约束条件
lb = np.array([1,10,1,2])  # 变量下界
ub = np.array([400,400,200,200])  # 变量上界

# 初始化粒子的位置和速度
positions = np.random.rand(num, 4) * (ub - lb) + lb
velocities = np.zeros((num, 4))

# 初始化个体最佳位置和适应度值
personal_best_positions = positions.copy()
personal_best_fitness = np.apply_along_axis(lambda row: fitness(*row), 1, positions)

# 初始化全局最佳位置和适应度值
global_best_fitness = np.max(personal_best_fitness)
global_best_position = personal_best_positions[np.argmax(personal_best_fitness)]

#以上time:22min

# PSO主循环
for iteration in range(max_iterations):
    # 更新粒子速度和位置
    inertia_weight = 0.9 - (0.9 - 0.4) * iteration / max_iterations  # 惯性权重逐渐减小
    cognitive_weight = 2.1
    social_weight = 2.1
    r1 = np.random.rand(num, 4)
    r2 = np.random.rand(num, 4)
    # 多样性维护参数
    diversity_weight = 0.1  # 调整这个参数来控制多样性维护的影响程度

    velocities = inertia_weight * velocities + \
        cognitive_weight * r1 * (personal_best_positions - positions) + \
        social_weight * r2 * (global_best_position - positions)

    # 引入多样性维护
    diversity_term = diversity_weight * (np.random.rand(num, 4) - 0.5) * (ub - lb)
    velocities = velocities + diversity_term

    positions = positions + velocities

    # 边界处理
    positions = np.maximum(positions, lb)
    positions = np.minimum(positions, ub)

    # 计算当前适应度值
    current_fitness = np.apply_along_axis(lambda row: fitness(*row), 1, positions)

    # 更新个体最佳位置和全局最佳位置
    update_indices = current_fitness > personal_best_fitness
    personal_best_positions[update_indices] = positions[update_indices]
    personal_best_fitness[update_indices] = current_fitness[update_indices]

    current_global_best_fitness = np.max(personal_best_fitness)
    if current_global_best_fitness > global_best_fitness:
        global_best_fitness = current_global_best_fitness
        global_best_position = personal_best_positions[np.argmax(personal_best_fitness)]

    # 显示当前迭代结果
    print(f'迭代次数 {iteration+1}, 随机森林最高F1值: {global_best_fitness}')
    fitnessbest.append(global_best_fitness)

# 绘制出每次迭代最佳适应度的变化图
plt.figure()
plt.plot(fitnessbest)
plt.xlabel('迭代次数')
plt.ylabel('适应度值')
#plt.show()

int_global_best_position = [int(element) for element in global_best_position]

# 输出最终结果
print('运行完成!')
print(f'最优参数: {int_global_best_position}')
print(f'最高F1值: {global_best_fitness}')