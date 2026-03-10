import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 读取清洗后的数据
df = pd.read_csv(r'd:\Code\data-analysis\cleaned_data.csv')

print("=" * 80)
print("机器学习分析模块")
print("=" * 80)

# ==================== 1. 数据预处理 ====================
print("\n1. 数据预处理")

# 选择用于分析的特征
feature_cols = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                'OrderCount', 'OrderAmountHikeFromlastYear', 'DaySinceLastOrder',
                'NumberOfStreamerFollowed', 'SatisfactionScore', 'CouponUsed',
                'DiscountAmount']

# 分类变量处理
cat_cols = ['PreferredLoginDevice', 'MaritalStatus', 'Gender', 'PreferedOrderCat']
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    feature_cols.append(col + '_encoded')

# 目标变量
target = 'Churn'

# 特征标准化
scaler = StandardScaler()
X = df[feature_cols]
y = df[target]
X_scaled = scaler.fit_transform(X)

print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(df)}")

# ==================== 2. 客户分群 ====================
print("\n2. 客户分群分析")

# 使用K-means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 分析每个集群的特征
cluster_analysis = df.groupby('Cluster').agg({
    'Tenure': 'mean',
    'OrderCount': 'mean',
    'HourSpendOnApp': 'mean',
    'SatisfactionScore': 'mean',
    'Churn': 'mean',
    'CustomerID': 'count'
}).round(2)

cluster_analysis.columns = ['平均使用时长', '平均订单数', '平均APP使用时长', 
                           '平均满意度', '流失率', '用户数']

print("\nK-means聚类结果:")
print(cluster_analysis)

# 集群命名
cluster_names = {
    0: '高价值忠诚用户',
    1: '新用户',
    2: '普通活跃用户',
    3: '低价值用户'
}

df['ClusterName'] = df['Cluster'].map(cluster_names)

print("\n集群命名:")
for cluster_id, name in cluster_names.items():
    count = len(df[df['Cluster'] == cluster_id])
    churn_rate = df[df['Cluster'] == cluster_id]['Churn'].mean() * 100
    print(f"  集群{cluster_id}: {name} ({count}人, 流失率: {churn_rate:.1f}%)")

# ==================== 3. 流失预测模型 ====================
print("\n3. 流失预测模型")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.3, 
                                                    random_state=42)

print(f"训练集: {len(X_train)}样本")
print(f"测试集: {len(X_test)}样本")

# 模型1: 随机森林
print("\n3.1 随机森林模型")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_rf))

# 特征重要性
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(feature_importance.head(10))

# 模型2: 梯度提升
print("\n3.2 梯度提升模型")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred_gb):.4f}")

# 模型3: 逻辑回归
print("\n3.3 逻辑回归模型")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred_lr):.4f}")

# ==================== 4. 模型评估 ====================
print("\n4. 模型性能比较")

models = {
    '随机森林': rf,
    '梯度提升': gb,
    '逻辑回归': lr
}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name}: 准确率 = {accuracy:.4f}")

# 选择最佳模型
best_model = max(models.items(), key=lambda x: accuracy_score(y_test, x[1].predict(X_test)))
print(f"\n最佳模型: {best_model[0]}")

# ==================== 5. 预测新用户流失概率 ====================
print("\n5. 流失概率预测")

# 为每个用户计算流失概率
df['ChurnProbability'] = best_model[1].predict_proba(X_scaled)[:, 1]

# 风险等级划分
def risk_level(prob):
    if prob < 0.2:
        return '低风险'
    elif prob < 0.5:
        return '中风险'
    else:
        return '高风险'

df['RiskLevel'] = df['ChurnProbability'].apply(risk_level)

risk_distribution = df['RiskLevel'].value_counts()
print("\n风险等级分布:")
print(risk_distribution)

# ==================== 6. 结果保存 ====================
print("\n6. 保存分析结果")

# 保存带有预测结果的数据
df.to_csv(r'd:\Code\data-analysis\ml_results.csv', index=False)
print("预测结果已保存至: ml_results.csv")

# 保存模型
import joblib
joblib.dump(best_model[1], r'd:\Code\data-analysis\churn_model.pkl')
joblib.dump(scaler, r'd:\Code\data-analysis\scaler.pkl')
print("模型已保存至: churn_model.pkl 和 scaler.pkl")

print("\n" + "=" * 80)
print("机器学习分析完成！")
print("=" * 80)
