import matplotlib.font_manager
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - 支持中英文显示

# 明确指定SimHei字体
font_path = 'C:/Windows/Fonts/simhei.ttf'
if os.path.exists(font_path):
    font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['font.family'] = 'sans-serif'
else:
    # 如果SimHei不存在，尝试使用Microsoft YaHei
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    if os.path.exists(font_path):
        font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei',
                                           'DejaVu Sans', 'Arial']
        plt.rcParams['font.family'] = 'sans-serif'
    else:
        # 回退到默认字体
        font_prop = {'family': 'sans-serif'}

plt.rcParams['axes.unicode_minus'] = False

# 为Seaborn设置字体
sns.set(font=font_prop.get_name() if hasattr(
    font_prop, 'get_name') else 'sans-serif')
sns.set_style("whitegrid")

# 读取清洗后的数据
df = pd.read_csv(r'd:\Code\data-analysis\cleaned_data.csv')

# 创建输出目录
output_dir = r'd:\Code\data-analysis\charts'
os.makedirs(output_dir, exist_ok=True)

# 定义颜色方案
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
churn_colors = ['#2E86AB', '#C73E1D']

# 中文标签映射
labels_cn = {
    'Retained Users': '留存用户',
    'Churned Users': '流失用户',
    'Number of Users': '用户数',
    'User Churn Distribution': '用户流失分布',
    'Retained vs Churned Users': '留存 vs 流失用户数',
    'User Segment Distribution': '用户分层分布',
    'Churn Rate (%)': '流失率 (%)',
    'Churn Rate by User Segment': '各用户分层流失率',
    'Retained': '留存用户',
    'Churned': '流失用户',
    'Retained vs Churned Users Comparison': '留存用户 vs 流失用户 对比',
    'Tenure (Months)': '使用时长(月)',
    'Average Tenure': '平均使用时长',
    'Order Count': '订单数',
    'Average Orders': '平均订单数',
    'Satisfaction Score': '满意度评分',
    'Average Satisfaction': '平均满意度',
    'Discount Amount': '折扣金额',
    'Average Discount': '平均折扣金额',
    'Churn Rate by Login Device': '登录设备与流失率',
    'Churn Rate by City Tier': '城市等级与流失率',
    'Churn Rate by Marital Status': '婚姻状况与流失率',
    'Churn Rate by Age Group': '年龄段与流失率',
    'Age Group': '年龄段',
    'Churn Rate by Order Category': '订单类别与流失率',
    'Churn Rate by Complaint Status': '投诉与流失率',
    'No Complaint': '无投诉',
    'Has Complaint': '有投诉',
    'User Distribution by Satisfaction Score': '各满意度评分的用户分布',
    'Satisfaction Score': '满意度评分',
    'Percentage (%)': '占比 (%)',
    'Churn Rate by Satisfaction Score': '各满意度评分的流失率',
    'Feature Correlation Heatmap': '特征相关性热力图',
    'Tenure (Months)': '使用时长(月)',
    'Number of Users': '用户数',
    'Tenure Distribution Comparison': '使用时长分布对比',
    'Tenure Group': '使用时长分组',
    'Churn Rate by Tenure Group': '各使用时长段的流失率',
    'E-Commerce User Churn Analysis Dashboard': '电商用户流失分析仪表盘',
    'Key Metrics': '关键业务指标',
    'Total Users': '总用户数',
    'Churn Rate': '流失率',
    'Avg Tenure': '平均使用时长',
    'Avg Orders': '平均订单数',
    'Avg Satisfaction': '平均满意度',
    'Complaint Rate': '投诉率',
    'Churn Rate by Segment': '各分层流失率',
    'Complaint Impact on Churn': '投诉对流失的影响',
    'Churn Rate by Tenure': '使用时长与流失率',
    'Churn Rate by Category': '订单类别流失率',
    'Key Findings': '主要发现',
    '0-3M': '0-3月',
    '3-6M': '3-6月',
    '6-12M': '6-12月',
    '12-24M': '12-24月',
    '24M+': '24月+',
    'Tier 1': '一线城市',
    'Tier 2': '二线城市',
    'Tier 3': '三线城市',
    'Single': '单身',
    'Married': '已婚',
    'Divorced': '离异',
    'Mobile Phone': '手机',
    'Phone': '电话',
    'Pad': '平板',
    'Grocery': '杂货',
    'Fashion': '时尚',
    'Household': '家居',
    'Laptop & Accessory': '笔记本及配件',
    'Others': '其他',
    '高价值用户': 'High Value',
    '中等价值用户': 'Medium Value',
    '普通用户': 'Regular',
    '低价值用户': 'Low Value'
}

print("正在生成可视化图表...")

# ==================== 图1: 用户流失概览 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

churn_counts = df['Churn'].value_counts()
labels = ['留存用户', '流失用户']
sizes = [churn_counts[0], churn_counts[1]]
explode = (0, 0.05)

axes[0].pie(sizes, explode=explode, labels=labels, colors=churn_colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontproperties': font_prop})
axes[0].set_title('用户流失分布', fontsize=14, fontweight='bold',
                  fontproperties=font_prop)


bars = axes[1].bar(labels, sizes, color=churn_colors,
                   edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('用户数', fontsize=12, fontproperties=font_prop)
axes[1].set_title('留存 vs 流失用户数', fontsize=14,
                  fontweight='bold', fontproperties=font_prop)
# 设置x轴标签字体
axes[1].set_xticklabels(['留存用户', '流失用户'], fontproperties=font_prop)
for bar, size in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                 fontproperties=font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_churn_overview.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图1: 用户流失概览 - 已生成")

# ==================== 图2: 用户分层分析 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

segment_counts = df['UserSegment'].value_counts()
axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
            colors=colors[:len(segment_counts)], startangle=90,
            textprops={'fontsize': 11, 'fontproperties': font_prop})
axes[0].set_title('用户分层分布', fontsize=14, fontweight='bold',
                  fontproperties=font_prop)


segment_churn = df.groupby('UserSegment')[
    'Churn'].mean().sort_values(ascending=True)
bars = axes[1].barh(segment_churn.index, segment_churn.values * 100,
                    color=['#2E86AB' if x < 0.15 else '#F18F01' if x < 0.20 else '#C73E1D'
                           for x in segment_churn.values])
axes[1].set_xlabel('流失率 (%)', fontsize=12, fontproperties=font_prop)
axes[1].set_title('各用户分层流失率', fontsize=14, fontweight='bold',
                  fontproperties=font_prop)
# 设置y轴标签字体
axes[1].set_yticklabels(segment_churn.index, fontproperties=font_prop)
for i, (idx, val) in enumerate(segment_churn.items()):
    axes[1].text(val * 100 + 0.5, i, f'{val*100:.1f}%',
                 va='center', fontsize=10, fontweight='bold',
                 fontproperties=font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_user_segmentation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图2: 用户分层分析 - 已生成")

# ==================== 图3: 流失用户 vs 留存用户对比 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

comparison_metrics = {
    'Tenure': ('使用时长(月)', '平均使用时长'),
    'OrderCount': ('订单数', '平均订单数'),
    'SatisfactionScore': ('满意度评分', '平均满意度'),
    'DiscountAmount': ('折扣金额', '平均折扣金额')
}

for idx, (col, (xlabel, title)) in enumerate(comparison_metrics.items()):
    ax = axes[idx // 2, idx % 2]

    data_to_plot = [df[df['Churn'] == 0][col], df[df['Churn'] == 1][col]]
    bp = ax.boxplot(data_to_plot, labels=['留存用户', '流失用户'],
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#C73E1D')

    ax.set_ylabel(xlabel, fontsize=11, fontproperties=font_prop)
    ax.set_title(title, fontsize=12, fontweight='bold',
                 fontproperties=font_prop)
    # 设置x轴标签字体
    ax.set_xticklabels(['留存用户', '流失用户'], fontproperties=font_prop)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('留存用户 vs 流失用户 关键指标对比', fontsize=14,
             fontweight='bold', y=1.02, fontproperties=font_prop)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_user_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图3: 用户对比分析 - 已生成")

# ==================== 图4: 各维度流失率分析 ====================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. 登录设备
device_churn = df.groupby('PreferredLoginDevice')['Churn'].mean().sort_values()
device_labels = [labels_cn.get(x, x) for x in device_churn.index]
axes[0, 0].bar(device_labels, device_churn.values *
               100, color=colors[:len(device_churn)])
axes[0, 0].set_title('登录设备与流失率', fontsize=12,
                     fontweight='bold', fontproperties=font_prop)
axes[0, 0].set_ylabel('流失率 (%)', fontproperties=font_prop)
# 设置x轴标签字体
axes[0, 0].set_xticklabels(
    device_labels, fontproperties=font_prop, rotation=45)


# 2. 城市等级
city_churn = df.groupby('CityTier')['Churn'].mean()
city_labels = ['一线城市', '二线城市', '三线城市']
axes[0, 1].bar(city_labels, city_churn.values * 100,
               color=['#2E86AB', '#F18F01', '#C73E1D'])
axes[0, 1].set_title('城市等级与流失率', fontsize=12,
                     fontweight='bold', fontproperties=font_prop)
axes[0, 1].set_ylabel('流失率 (%)', fontproperties=font_prop)
# 设置x轴标签字体
axes[0, 1].set_xticklabels(city_labels, fontproperties=font_prop)


# 3. 婚姻状况
marital_churn = df.groupby('MaritalStatus')['Churn'].mean().sort_values()
marital_labels = [labels_cn.get(x, x) for x in marital_churn.index]
axes[0, 2].bar(marital_labels, marital_churn.values * 100,
               color=colors[:len(marital_churn)])
axes[0, 2].set_title('婚姻状况与流失率', fontsize=12,
                     fontweight='bold', fontproperties=font_prop)
axes[0, 2].set_ylabel('流失率 (%)', fontproperties=font_prop)
# 设置x轴标签字体
axes[0, 2].set_xticklabels(
    marital_labels, fontproperties=font_prop, rotation=45)


# 4. 年龄段
age_churn = df.groupby('AgeGroup')['Churn'].mean()
axes[1, 0].plot(age_churn.index, age_churn.values * 100, marker='o', linewidth=2,
                markersize=8, color='#C73E1D')
axes[1, 0].set_title('年龄段与流失率', fontsize=12,
                     fontweight='bold', fontproperties=font_prop)
axes[1, 0].set_xlabel('年龄段', fontproperties=font_prop)
axes[1, 0].set_ylabel('流失率 (%)', fontproperties=font_prop)
# 设置x轴标签字体
axes[1, 0].set_xticklabels(age_churn.index, fontproperties=font_prop)
axes[1, 0].grid(alpha=0.3)


# 5. 订单类别
cat_churn = df.groupby('PreferedOrderCat')['Churn'].mean().sort_values()
cat_labels = [labels_cn.get(x, x) for x in cat_churn.index]
axes[1, 1].barh(cat_labels, cat_churn.values * 100,
                color=['#2E86AB' if x < 0.15 else '#F18F01' if x < 0.20 else '#C73E1D'
                       for x in cat_churn.values])
axes[1, 1].set_title('订单类别与流失率', fontsize=12,
                     fontweight='bold', fontproperties=font_prop)
axes[1, 1].set_xlabel('流失率 (%)', fontproperties=font_prop)
# 设置y轴标签字体
axes[1, 1].set_yticklabels(cat_labels, fontproperties=font_prop)


# 6. 投诉情况
complain_churn = df.groupby('Complain')['Churn'].mean()
complain_labels = ['无投诉', '有投诉']
bars = axes[1, 2].bar(complain_labels, complain_churn.values * 100,
                      color=['#2E86AB', '#C73E1D'])
axes[1, 2].set_title('投诉与流失率', fontsize=12,
                     fontweight='bold', fontproperties=font_prop)
axes[1, 2].set_ylabel('流失率 (%)', fontproperties=font_prop)
# 设置x轴标签字体
axes[1, 2].set_xticklabels(complain_labels, fontproperties=font_prop)
for bar, val in zip(bars, complain_churn.values):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold',
                    fontproperties=font_prop)

plt.suptitle('各维度流失率分析', fontsize=14, fontweight='bold',
             y=1.02, fontproperties=font_prop)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_churn_by_dimensions.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图4: 各维度流失率分析 - 已生成")

# ==================== 图5: 满意度与流失关系 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

satisfaction_dist = df.groupby(
    ['SatisfactionScore', 'Churn']).size().unstack(fill_value=0)
satisfaction_pct = satisfaction_dist.div(
    satisfaction_dist.sum(axis=1), axis=0) * 100

satisfaction_pct.plot(kind='bar', stacked=True,
                      ax=axes[0], color=churn_colors, width=0.7)
axes[0].set_title('各满意度评分的用户分布', fontsize=12,
                  fontweight='bold', fontproperties=font_prop)
axes[0].set_xlabel('满意度评分', fontproperties=font_prop)
axes[0].set_ylabel('占比 (%)', fontproperties=font_prop)
# 为图例设置字体
legend = axes[0].legend(['留存', '流失'], loc='upper left')
for text in legend.get_texts():
    text.set_fontproperties(font_prop)
axes[0].tick_params(axis='x', rotation=0)


satisfaction_churn = df.groupby('SatisfactionScore')['Churn'].mean()
bars = axes[1].bar(satisfaction_churn.index, satisfaction_churn.values * 100,
                   color=['#2E86AB' if x < 0.15 else '#F18F01' if x < 0.20 else '#C73E1D'
                          for x in satisfaction_churn.values])
axes[1].set_title('各满意度评分的流失率', fontsize=12,
                  fontweight='bold', fontproperties=font_prop)
axes[1].set_xlabel('满意度评分', fontproperties=font_prop)
axes[1].set_ylabel('流失率 (%)', fontproperties=font_prop)
for bar, val in zip(bars, satisfaction_churn.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                 fontproperties=font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/05_satisfaction_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图5: 满意度分析 - 已生成")

# ==================== 图6: 相关性热力图 ====================
fig, ax = plt.subplots(figsize=(12, 10))

correlation_cols = ['Churn', 'Tenure', 'HourSpendOnApp', 'OrderCount',
                    'OrderAmountHikeFromlastYear', 'DaySinceLastOrder',
                    'SatisfactionScore', 'Complain', 'CouponUsed', 'DiscountAmount',
                    'CityTier', 'NumberOfStreamerFollowed']

corr_matrix = df[correlation_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)
ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold',
             pad=20, fontproperties=font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/06_correlation_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图6: 相关性热力图 - 已生成")

# ==================== 图7: 使用时长与流失关系 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df[df['Churn'] == 0]['Tenure'], bins=30, alpha=0.7, label='留存用户',
             color='#2E86AB', edgecolor='black')
axes[0].hist(df[df['Churn'] == 1]['Tenure'], bins=30, alpha=0.7, label='流失用户',
             color='#C73E1D', edgecolor='black')
axes[0].set_xlabel('使用时长 (月)', fontsize=11, fontproperties=font_prop)
axes[0].set_ylabel('用户数', fontsize=11, fontproperties=font_prop)
axes[0].set_title('使用时长分布对比', fontsize=12, fontweight='bold',
                  fontproperties=font_prop)
# 为图例设置字体
legend = axes[0].legend()
for text in legend.get_texts():
    text.set_fontproperties(font_prop)
axes[0].grid(alpha=0.3)


df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 3, 6, 12, 24, 100],
                           labels=['0-3月', '3-6月', '6-12月', '12-24月', '24月+'])
tenure_churn = df.groupby('TenureGroup')['Churn'].mean()
bars = axes[1].bar(tenure_churn.index, tenure_churn.values * 100,
                   color=['#C73E1D', '#F18F01', '#6A994E', '#2E86AB', '#2E86AB'])
axes[1].set_xlabel('使用时长分组', fontsize=11, fontproperties=font_prop)
axes[1].set_ylabel('流失率 (%)', fontsize=11, fontproperties=font_prop)
axes[1].set_title('各使用时长段的流失率', fontsize=12,
                  fontweight='bold', fontproperties=font_prop)
# 设置x轴标签字体
axes[1].set_xticklabels(
    tenure_churn.index, fontproperties=font_prop, rotation=15)
for bar, val in zip(bars, tenure_churn.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold',
                 fontproperties=font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/07_tenure_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图7: 使用时长分析 - 已生成")

# ==================== 图8: 综合仪表盘 ====================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('电商用户流失分析仪表盘', fontsize=18, fontweight='bold',
             y=0.98, fontproperties=font_prop)


# 1. 流失概览
ax1 = fig.add_subplot(gs[0, 0])
churn_counts = df['Churn'].value_counts()
ax1.pie(churn_counts.values, labels=['留存', '流失'], colors=churn_colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontproperties': font_prop})
ax1.set_title('用户流失分布', fontsize=12, fontweight='bold',
              fontproperties=font_prop)


# 2. 关键指标
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
metrics_text = f"""关键业务指标

总用户数: {len(df):,}
流失率: {df['Churn'].mean()*100:.1f}%
平均使用时长: {df['Tenure'].mean():.1f}月
平均订单数: {df['OrderCount'].mean():.1f}单
平均满意度: {df['SatisfactionScore'].mean():.1f}分
投诉率: {df['Complain'].mean()*100:.1f}%
"""
ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontproperties=font_prop)


# 3. 用户分层
ax3 = fig.add_subplot(gs[0, 2])
segment_churn = df.groupby('UserSegment')['Churn'].mean().sort_values()
bars = ax3.barh(segment_churn.index, segment_churn.values * 100,
                color=['#2E86AB', '#6A994E', '#F18F01', '#C73E1D'])
ax3.set_xlabel('流失率 (%)', fontproperties=font_prop)
ax3.set_title('各分层流失率', fontsize=12, fontweight='bold',
              fontproperties=font_prop)
# 设置y轴标签字体
ax3.set_yticklabels(segment_churn.index, fontproperties=font_prop)


# 4. 投诉影响
ax4 = fig.add_subplot(gs[1, 0])
complain_churn = df.groupby('Complain')['Churn'].mean()
bars = ax4.bar(['无投诉', '有投诉'], complain_churn.values * 100,
               color=['#2E86AB', '#C73E1D'])
ax4.set_ylabel('流失率 (%)', fontproperties=font_prop)
ax4.set_title('投诉对流失的影响', fontsize=12, fontweight='bold',
              fontproperties=font_prop)
# 设置x轴标签字体
ax4.set_xticklabels(['无投诉', '有投诉'], fontproperties=font_prop)
for bar, val in zip(bars, complain_churn.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
             fontproperties=font_prop)


# 5. 使用时长趋势
ax5 = fig.add_subplot(gs[1, 1])
tenure_churn = df.groupby('TenureGroup')['Churn'].mean()
ax5.plot(range(len(tenure_churn)), tenure_churn.values * 100, marker='o',
         linewidth=2, markersize=8, color='#C73E1D')
ax5.set_xticks(range(len(tenure_churn)))
ax5.set_xticklabels(tenure_churn.index, rotation=15, fontproperties=font_prop)
ax5.set_ylabel('流失率 (%)', fontproperties=font_prop)
ax5.set_title('使用时长与流失率', fontsize=12, fontweight='bold',
              fontproperties=font_prop)
ax5.grid(alpha=0.3)


# 6. 订单类别
ax6 = fig.add_subplot(gs[1, 2])
cat_churn = df.groupby('PreferedOrderCat')['Churn'].mean().sort_values()
cat_labels = [labels_cn.get(x, x) for x in cat_churn.index]
ax6.barh(cat_labels, cat_churn.values * 100,
         color=['#2E86AB' if x < 0.15 else '#F18F01' if x < 0.20 else '#C73E1D'
                for x in cat_churn.values])
ax6.set_xlabel('流失率 (%)', fontproperties=font_prop)
ax6.set_title('订单类别流失率', fontsize=12, fontweight='bold',
              fontproperties=font_prop)
# 设置y轴标签字体
ax6.set_yticklabels(cat_labels, fontproperties=font_prop)


# 7. 满意度
ax7 = fig.add_subplot(gs[2, :2])
satisfaction_churn = df.groupby('SatisfactionScore')['Churn'].mean()
bars = ax7.bar(satisfaction_churn.index, satisfaction_churn.values * 100,
               color=['#2E86AB' if x < 0.15 else '#F18F01' if x < 0.20 else '#C73E1D'
                      for x in satisfaction_churn.values])
ax7.set_xlabel('满意度评分', fontproperties=font_prop)
ax7.set_ylabel('流失率 (%)', fontproperties=font_prop)
ax7.set_title('满意度与流失率关系', fontsize=12,
              fontweight='bold', fontproperties=font_prop)
# 设置x轴标签字体
ax7.set_xticklabels(satisfaction_churn.index, fontproperties=font_prop)
for bar, val in zip(bars, satisfaction_churn.values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
             fontproperties=font_prop)


# 8. 流失原因总结
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
summary_text = """主要发现

• 新用户流失严重
  (0-3月流失率最高)

• 投诉用户流失率高
  (31.7% vs 10.9%)

• 满意度5分用户
  反而流失率高
  
• 手机订单类别
  流失率最高(27.5%)
"""
ax8.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
         fontproperties=font_prop)

plt.savefig(f'{output_dir}/08_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 图8: 综合仪表盘 - 已生成")

print("\n" + "=" * 60)
print("所有可视化图表已生成完毕！")
print(f"图表保存位置: {output_dir}")
print("=" * 60)
print("\n生成的图表列表:")
for i, file in enumerate(sorted(os.listdir(output_dir)), 1):
    if file.endswith('.png'):
        print(f"  {i}. {file}")
