import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 读取数据
file_path = r'd:\Code\data-analysis\Project Dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='E Comm')

print("=" * 80)
print("电商用户流失分析 - 综合数据分析报告")
print("=" * 80)

# ==================== 1. 数据概览 ====================
print("\n" + "=" * 80)
print("一、数据概览")
print("=" * 80)
print(f"\n总用户数: {df.shape[0]:,}")
print(f"特征数量: {df.shape[1]} 个")
print(f"\n流失率: {df['Churn'].mean()*100:.2f}%")
print(f"留存用户数: {(df['Churn']==0).sum():,}")
print(f"流失用户数: {(df['Churn']==1).sum():,}")

# ==================== 2. 数据清洗 ====================
print("\n" + "=" * 80)
print("二、数据清洗与预处理")
print("=" * 80)

# 2.1 缺失值处理
print("\n【缺失值分析】")
missing_cols = df.columns[df.isnull().any()].tolist()
print(f"存在缺失值的列: {missing_cols}")

# 填充缺失值
df_clean = df.copy()
# 数值型用中位数填充
num_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderCount', 
            'OrderAmountHikeFromlastYear', 'DaySinceLastOrder', 'CouponUsed']
for col in num_cols:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

print("\n缺失值已处理，使用各列中位数填充")
print(f"清洗后缺失值总数: {df_clean.isnull().sum().sum()}")

# 2.2 异常值检测
print("\n【异常值检测】")
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'CustomerID':
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_clean[(df_clean[col] < Q1 - 1.5*IQR) | (df_clean[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} 个异常值 ({len(outliers)/len(df_clean)*100:.1f}%)")

# ==================== 3. 描述性统计分析 ====================
print("\n" + "=" * 80)
print("三、描述性统计分析")
print("=" * 80)

# 3.1 数值型变量统计
print("\n【数值型变量统计】")
numeric_summary = df_clean[numeric_cols].describe()
print(numeric_summary.round(2))

# 3.2 分类型变量统计
print("\n【分类型变量统计】")
categorical_cols = ['PreferredLoginDevice', 'MaritalStatus', 'Gender', 'PreferedOrderCat']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df_clean[col].value_counts().head(10))

# ==================== 4. 用户行为分析 ====================
print("\n" + "=" * 80)
print("四、用户行为分析")
print("=" * 80)

# 4.1 流失用户 vs 留存用户对比
print("\n【流失用户 vs 留存用户对比】")
comparison_metrics = ['Tenure', 'HourSpendOnApp', 'OrderCount', 
                      'OrderAmountHikeFromlastYear', 'DaySinceLastOrder',
                      'SatisfactionScore', 'CouponUsed', 'DiscountAmount']

comparison = df_clean.groupby('Churn')[comparison_metrics].mean()
comparison.index = ['留存用户', '流失用户']
print(comparison.round(2))

# 4.2 关键指标分析
print("\n【关键业务指标】")
print(f"平均使用时长: {df_clean['Tenure'].mean():.1f} 个月")
print(f"平均APP使用时长: {df_clean['HourSpendOnApp'].mean():.2f} 小时")
print(f"平均订单数: {df_clean['OrderCount'].mean():.2f} 单")
print(f"平均满意度: {df_clean['SatisfactionScore'].mean():.2f} 分")
print(f"投诉率: {df_clean['Complain'].mean()*100:.2f}%")
print(f"优惠券使用率: {df_clean['CouponUsed'].mean()*100:.2f}%")

# 4.3 用户分层分析
print("\n【用户分层分析 - RFM模型】")
# 使用Tenure, OrderCount, OrderAmountHikeFromlastYear作为RFM替代指标
df_clean['R_Score'] = pd.qcut(df_clean['DaySinceLastOrder'], 5, labels=[5,4,3,2,1])
df_clean['F_Score'] = pd.qcut(df_clean['OrderCount'].rank(method='first'), 5, labels=[1,2,3,4,5])
df_clean['M_Score'] = pd.qcut(df_clean['DiscountAmount'].rank(method='first'), 5, labels=[1,2,3,4,5])

df_clean['RFM_Score'] = df_clean['R_Score'].astype(str) + \
                        df_clean['F_Score'].astype(str) + \
                        df_clean['M_Score'].astype(str)

# 用户价值分层
def segment_users(row):
    if row['F_Score'] >= 4 and row['M_Score'] >= 4:
        return '高价值用户'
    elif row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return '中等价值用户'
    elif row['F_Score'] >= 2:
        return '普通用户'
    else:
        return '低价值用户'

df_clean['UserSegment'] = df_clean.apply(segment_users, axis=1)
segment_analysis = df_clean.groupby('UserSegment').agg({
    'CustomerID': 'count',
    'Churn': 'mean'
}).round(3)
segment_analysis.columns = ['用户数', '流失率']
segment_analysis['占比'] = (segment_analysis['用户数'] / len(df_clean) * 100).round(1)
print(segment_analysis)

# ==================== 5. 流失原因分析 ====================
print("\n" + "=" * 80)
print("五、流失原因分析")
print("=" * 80)

# 5.1 各维度流失率分析
print("\n【各维度流失率分析】")

# 登录设备
print("\n1. 登录设备与流失率:")
device_churn = df_clean.groupby('PreferredLoginDevice')['Churn'].agg(['count', 'mean']).round(3)
device_churn.columns = ['用户数', '流失率']
print(device_churn)

# 城市等级
print("\n2. 城市等级与流失率:")
city_churn = df_clean.groupby('CityTier')['Churn'].agg(['count', 'mean']).round(3)
city_churn.columns = ['用户数', '流失率']
print(city_churn)

# 婚姻状况
print("\n3. 婚姻状况与流失率:")
marital_churn = df_clean.groupby('MaritalStatus')['Churn'].agg(['count', 'mean']).round(3)
marital_churn.columns = ['用户数', '流失率']
print(marital_churn)

# 年龄段
print("\n4. 年龄段与流失率:")
age_churn = df_clean.groupby('AgeGroup')['Churn'].agg(['count', 'mean']).round(3)
age_churn.columns = ['用户数', '流失率']
print(age_churn)

# 性别
print("\n5. 性别与流失率:")
gender_churn = df_clean.groupby('Gender')['Churn'].agg(['count', 'mean']).round(3)
gender_churn.columns = ['用户数', '流失率']
print(gender_churn)

# 订单类别
print("\n6. 订单类别与流失率:")
cat_churn = df_clean.groupby('PreferedOrderCat')['Churn'].agg(['count', 'mean']).round(3)
cat_churn.columns = ['用户数', '流失率']
print(cat_churn)

# 5.2 投诉与流失关系
print("\n【投诉与流失关系】")
complain_churn = df_clean.groupby('Complain')['Churn'].mean()
print(f"无投诉用户流失率: {complain_churn[0]*100:.2f}%")
print(f"有投诉用户流失率: {complain_churn[1]*100:.2f}%")

# 5.3 满意度与流失关系
print("\n【满意度与流失关系】")
satisfaction_churn = df_clean.groupby('SatisfactionScore')['Churn'].agg(['count', 'mean']).round(3)
satisfaction_churn.columns = ['用户数', '流失率']
print(satisfaction_churn)

# ==================== 6. 相关性分析 ====================
print("\n" + "=" * 80)
print("六、相关性分析")
print("=" * 80)

correlation_cols = ['Churn', 'Tenure', 'HourSpendOnApp', 'OrderCount', 
                    'OrderAmountHikeFromlastYear', 'DaySinceLastOrder',
                    'SatisfactionScore', 'Complain', 'CouponUsed', 'DiscountAmount']
corr_matrix = df_clean[correlation_cols].corr()['Churn'].sort_values(ascending=False)
print("\n与流失率的相关性（从高到低）:")
print(corr_matrix.round(3))

# 保存清洗后的数据
df_clean.to_csv(r'd:\Code\data-analysis\cleaned_data.csv', index=False)
print("\n\n清洗后的数据已保存至: cleaned_data.csv")

print("\n" + "=" * 80)
print("数据分析完成！")
print("=" * 80)
