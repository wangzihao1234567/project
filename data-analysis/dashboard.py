import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="电商用户分析仪表盘",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    /* 深色背景 */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        min-height: 100vh;
    }
    
    /* 侧边栏美化 */
    .sidebar {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        color: white !important;
        transition: all 0.3s ease;
    }
    
    .sidebar:hover {
        box-shadow: 2px 0 10px rgba(0,0,0,0.3);
    }
    
    .sidebar .sidebar-content {
        color: white !important;
        padding: 20px;
    }
    
    /* 侧边栏文本颜色 */
    .sidebar .stSelectbox label {
        color: white !important;
        font-weight: 500;
    }
    
    .sidebar .stSelectbox div {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    .sidebar .stSelectbox {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
    }
    
    .sidebar h1, .sidebar h2, .sidebar h3, .sidebar h4, .sidebar h5, .sidebar h6 {
        color: white !important;
    }
    
    .sidebar p, .sidebar span {
        color: white !important;
    }
    
    /* 标题样式 */
    h1, h2, h3, h4, h5 {
        color: #e0e0e0 !important;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    h1 {
        color: #4ecdc4 !important;
        font-weight: 700;
    }
    
    /* 正文文本颜色 */
    p, span, div {
        color: #e0e0e0 !important;
    }
    
    /* 指标卡片样式 */
    .stMetric {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(78, 205, 196, 0.3);
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-color: rgba(78, 205, 196, 0.6);
    }
    
    .stMetric label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stMetric value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ecdc4;
        transition: all 0.3s ease;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: #4ecdc4;
        color: #1a1a2e;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }
    
    .stButton > button:hover {
        background: #3cc9bf;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(78, 205, 196, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* 选择框和滑块样式 */
    .stSelectbox {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stSelectbox:hover {
        border-color: #3498db;
    }
    
    .stSlider {
        padding: 15px 0;
    }
    
    .stSlider > div {
        padding: 10px 0;
    }
    
    /* 图表容器样式 */
    .chart-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(78, 205, 196, 0.3);
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-color: rgba(78, 205, 196, 0.6);
    }
    
    /* 动画效果 */
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .animate-slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    .animate-scale-in {
        animation: scaleIn 0.4s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* 卡片样式 */
    .card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        padding: 25px;
        margin: 15px 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(78, 205, 196, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: #4ecdc4;
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-color: rgba(78, 205, 196, 0.6);
    }
    
    .card:hover::before {
        transform: scaleX(1);
    }
    
    /* 高亮样式 */
    .highlight {
        background: rgba(78, 205, 196, 0.2);
        border-radius: 8px;
        padding: 4px 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        color: #4ecdc4;
        border: 1px solid rgba(78, 205, 196, 0.4);
    }
    
    .highlight:hover {
        box-shadow: 0 2px 8px rgba(78, 205, 196, 0.4);
        background: rgba(78, 205, 196, 0.3);
    }
    
    /* 信息框样式 */
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.2);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    .info-box {
        background: rgba(25, 118, 210, 0.1);
        border-left: 4px solid #1976d2;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.2);
        transition: all 0.3s ease;
        border: 1px solid rgba(25, 118, 210, 0.3);
    }
    
    .success-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4caf50;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
        transition: all 0.3s ease;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: #4ecdc4;
        transition: width 1s ease-in-out;
    }
    
    /* 进度条容器样式 */
    .stProgress > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* 表格样式 */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(78, 205, 196, 0.3);
    }
    
    /* 表格文本颜色 */
    .stDataFrame th, .stDataFrame td {
        color: #e0e0e0 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    .stDataFrame th {
        background: rgba(78, 205, 196, 0.2) !important;
    }
    
    /* 滑块样式 */
    .stSlider > div > div > div {
        background: #4ecdc4;
    }
    
    /* 滑块轨道样式 */
    .stSlider > div > div {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* 加载动画 */
    .loader {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid #3498db;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* 页脚样式 */
    footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-top: 40px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 加载数据
df = pd.read_csv('ml_results.csv')

# 加载模型和标量
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# 页面标题
st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
st.title("🎯 电商用户分析仪表盘")
st.markdown("<p style='font-size: 1.1rem; color: #64748b;'>实时监控用户行为，智能预测流失风险</p>",
            unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# 顶部信息卡片
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='card animate-scale-in'>", unsafe_allow_html=True)
    st.subheader("📈 数据分析概览")
    st.markdown(
        f"<span style='font-size: 1.2rem;'>总用户数: <span class='highlight'>{len(df):,}</span></span>", unsafe_allow_html=True)
    st.markdown(
        f"<span style='font-size: 1.2rem;'>流失率: <span class='highlight'>{df['Churn'].mean()*100:.1f}%</span></span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 1.2rem;'>模型准确率: <span class='highlight'>94.02%</span></span>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card animate-slide-in'>", unsafe_allow_html=True)
    st.subheader("🎯 风险分布")
    high_risk = len(df[df['RiskLevel'] == '高风险'])
    medium_risk = len(df[df['RiskLevel'] == '中风险'])
    low_risk = len(df[df['RiskLevel'] == '低风险'])

    # 风险等级进度条
    total = len(df)
    st.markdown(
        f"<span style='font-weight: 500;'>高风险: **{high_risk:,}** ({high_risk/total*100:.1f}%)</span>", unsafe_allow_html=True)
    st.progress(high_risk/total, text=f"高风险用户 ({high_risk:,})")
    st.markdown(
        f"<span style='font-weight: 500;'>中风险: **{medium_risk:,}** ({medium_risk/total*100:.1f}%)</span>", unsafe_allow_html=True)
    st.progress(medium_risk/total, text=f"中风险用户 ({medium_risk:,})")
    st.markdown(
        f"<span style='font-weight: 500;'>低风险: **{low_risk:,}** ({low_risk/total*100:.1f}%)</span>", unsafe_allow_html=True)
    st.progress(low_risk/total, text=f"低风险用户 ({low_risk:,})")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card animate-fade-in'>", unsafe_allow_html=True)
    st.subheader("📊 系统状态")
    st.markdown("<span style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 1.2rem;'>✅</span> 数据加载完成</span>", unsafe_allow_html=True)
    st.markdown("<span style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 1.2rem;'>✅</span> 模型加载成功</span>", unsafe_allow_html=True)
    st.markdown("<span style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 1.2rem;'>✅</span> 实时预测可用</span>", unsafe_allow_html=True)
    st.markdown("<span style='display: flex; align-items: center; gap: 10px;'><span style='font-size: 1.2rem;'>✅</span> 可视化图表就绪</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# 侧边栏
st.sidebar.header("数据分析选项")
analysis_type = st.sidebar.selectbox(
    "选择分析类型",
    ["数据概览", "用户分群", "流失预测", "特征分析", "交互式预测"]
)

# 数据概览
if analysis_type == "数据概览":
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    st.header("📊 数据概览")

    # 关键指标卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<div class='card animate-scale-in' style='animation-delay: 0.1s;'>", unsafe_allow_html=True)
        st.metric("总用户数", f"{len(df):,}")
        st.metric("流失率", f"{df['Churn'].mean()*100:.1f}%")
        st.metric("平均使用时长", f"{df['Tenure'].mean():.1f}月")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div class='card animate-scale-in' style='animation-delay: 0.2s;'>", unsafe_allow_html=True)
        st.metric("平均订单数", f"{df['OrderCount'].mean():.1f}单")
        st.metric("平均满意度", f"{df['SatisfactionScore'].mean():.1f}分")
        st.metric("投诉率", f"{df['Complain'].mean()*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            "<div class='card animate-scale-in' style='animation-delay: 0.3s;'>", unsafe_allow_html=True)
        st.metric("高风险用户", f"{len(df[df['RiskLevel']=='高风险']):,}")
        st.metric("中风险用户", f"{len(df[df['RiskLevel']=='中风险']):,}")
        st.metric("低风险用户", f"{len(df[df['RiskLevel']=='低风险']):,}")
        st.markdown("</div>", unsafe_allow_html=True)

    # 流失分布饼图 - 增加动画
    st.markdown("<div class='card animate-slide-in'>", unsafe_allow_html=True)
    st.subheader("👥 用户流失分布")
    churn_counts = df['Churn'].value_counts()
    fig = px.pie(
        names=['留存用户', '流失用户'],
        values=churn_counts.values,
        color=['#2E86AB', '#C73E1D'],
        title="用户流失分布",
        hole=0.3  # 环形图
    )
    fig.update_layout(
        title={'text': "用户流失分布", 'x': 0.5, 'xanchor': 'center'},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.1},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    # 添加动画效果
    fig.update_traces(
        hoverinfo='label+percent+value',
        textinfo='label+percent',
        textfont_size=14,
        marker=dict(
            line=dict(color='#ffffff', width=2)
        ),
        opacity=0.8,
        pull=[0.05, 0.1]  # 稍微拉出饼图扇区
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 风险等级分布 - 增加动画
    st.markdown("<div class='card animate-fade-in'>", unsafe_allow_html=True)
    st.subheader("⚡ 风险等级分布")
    risk_counts = df['RiskLevel'].value_counts()
    fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        color=risk_counts.index,
        title="风险等级分布",
        color_discrete_sequence=['#e74c3c', '#f39c12', '#27ae60']
    )
    fig.update_layout(
        title={'text': "风险等级分布", 'x': 0.5, 'xanchor': 'center'},
        xaxis={'title': '风险等级'},
        yaxis={'title': '用户数'},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    # 添加动画效果
    fig.update_traces(
        hoverinfo='x+y',
        marker=dict(
            opacity=0.8,
            line=dict(color='#ffffff', width=1)
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 使用时长分布 - 新增图表
    st.markdown("<div class='card animate-scale-in'>", unsafe_allow_html=True)
    st.subheader("📅 使用时长分布")
    fig = px.histogram(
        df,
        x='Tenure',
        nbins=20,
        title="用户使用时长分布",
        color_discrete_sequence=['#3498db']
    )
    fig.update_layout(
        title={'text': "用户使用时长分布", 'x': 0.5, 'xanchor': 'center'},
        xaxis={'title': '使用时长 (月)'},
        yaxis={'title': '用户数'},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# 用户分群
elif analysis_type == "用户分群":
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    st.header("👥 用户分群分析")

    cluster_stats = df.groupby('ClusterName').agg({
        'CustomerID': 'count',
        'Churn': 'mean',
        'Tenure': 'mean',
        'OrderCount': 'mean',
        'SatisfactionScore': 'mean'
    }).round(2)

    cluster_stats.columns = ['用户数', '流失率', '平均使用时长', '平均订单数', '平均满意度']

    # 集群统计表格
    st.markdown("<div class='card animate-scale-in'>", unsafe_allow_html=True)
    st.subheader("📋 集群统计")
    st.dataframe(cluster_stats.style.background_gradient(cmap='viridis'))
    st.markdown("</div>", unsafe_allow_html=True)

    # 集群用户数和流失率对比
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div class='card animate-slide-in' style='animation-delay: 0.1s;'>", unsafe_allow_html=True)
        st.subheader("👥 各集群用户数")
        fig = px.bar(
            x=cluster_stats.index,
            y=cluster_stats['用户数'],
            color=cluster_stats.index,
            title="各集群用户数",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            title={'text': "各集群用户数", 'x': 0.5, 'xanchor': 'center'},
            xaxis={'title': '用户群体'},
            yaxis={'title': '用户数'},
            transition={'duration': 1000},
            template="plotly_white",
            font={'family': 'SimHei, Noto Sans SC, sans-serif'}
        )
        # 添加动画效果
        fig.update_traces(
            hoverinfo='x+y',
            marker=dict(
                opacity=0.8,
                line=dict(color='#ffffff', width=1)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div class='card animate-slide-in' style='animation-delay: 0.2s;'>", unsafe_allow_html=True)
        st.subheader("📉 各集群流失率")
        fig = px.bar(
            x=cluster_stats.index,
            y=cluster_stats['流失率']*100,
            color=cluster_stats.index,
            title="各集群流失率 (%)",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(
            title={'text': "各集群流失率", 'x': 0.5, 'xanchor': 'center'},
            xaxis={'title': '用户群体'},
            yaxis={'title': '流失率 (%)'},
            transition={'duration': 1000},
            template="plotly_white",
            font={'family': 'SimHei, Noto Sans SC, sans-serif'}
        )
        # 添加动画效果
        fig.update_traces(
            hoverinfo='x+y',
            marker=dict(
                opacity=0.8,
                line=dict(color='#ffffff', width=1)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 集群特征雷达图 - 增加动画
    st.markdown("<div class='card animate-fade-in'>", unsafe_allow_html=True)
    st.subheader("📊 集群特征雷达图")
    radar_data = cluster_stats[['平均使用时长', '平均订单数', '平均满意度']]

    # 重新组织数据格式以适应px.line_polar
    radar_df = pd.DataFrame(columns=['Cluster', 'Feature', 'Value'])
    for cluster in radar_data.index:
        for feature in radar_data.columns:
            radar_df = pd.concat([radar_df, pd.DataFrame({
                'Cluster': [cluster],
                'Feature': [feature],
                'Value': [radar_data.loc[cluster, feature]]
            })], ignore_index=True)

    fig = px.line_polar(
        radar_df,
        r='Value',
        theta='Feature',
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title="各集群特征对比"
    )
    fig.update_layout(
        title={'text': "各集群特征对比", 'x': 0.5, 'xanchor': 'center'},
        polar={'radialaxis': {'visible': True}},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    # 添加动画效果
    fig.update_traces(
        hoverinfo='r+theta',
        line=dict(width=3),
        marker=dict(size=8)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 集群特征详细分析
    st.markdown("<div class='card animate-scale-in'>", unsafe_allow_html=True)
    st.subheader("🔍 集群特征详细分析")
    selected_cluster = st.selectbox("选择集群", cluster_stats.index)

    cluster_data = df[df['ClusterName'] == selected_cluster]
    st.markdown(
        f"<h4 style='color: #3498db;'>{selected_cluster} 详细信息</h4>", unsafe_allow_html=True)
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.markdown(
            f"<span style='font-size: 1.1rem;'>• 总用户数: <span class='highlight'>{len(cluster_data):,}</span></span>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='font-size: 1.1rem;'>• 流失率: <span class='highlight'>{cluster_data['Churn'].mean()*100:.1f}%</span></span>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='font-size: 1.1rem;'>• 平均使用时长: <span class='highlight'>{cluster_data['Tenure'].mean():.1f}</span> 月</span>", unsafe_allow_html=True)
    with col_stats2:
        st.markdown(
            f"<span style='font-size: 1.1rem;'>• 平均订单数: <span class='highlight'>{cluster_data['OrderCount'].mean():.1f}</span> 单</span>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='font-size: 1.1rem;'>• 平均满意度: <span class='highlight'>{cluster_data['SatisfactionScore'].mean():.1f}</span> 分</span>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='font-size: 1.1rem;'>• 平均折扣金额: <span class='highlight'>{cluster_data['DiscountAmount'].mean():.1f}</span> 元</span>", unsafe_allow_html=True)

    # 集群特征分布 - 新增图表
    st.subheader("📈 集群特征分布")
    feature_to_plot = st.selectbox(
        "选择特征", ['Tenure', 'OrderCount', 'SatisfactionScore', 'DiscountAmount'])
    fig = px.histogram(
        cluster_data,
        x=feature_to_plot,
        nbins=20,
        title=f"{selected_cluster} - {feature_to_plot} 分布",
        color_discrete_sequence=['#9b59b6']
    )
    fig.update_layout(
        title={'text': f"{selected_cluster} - {feature_to_plot} 分布",
               'x': 0.5, 'xanchor': 'center'},
        xaxis={'title': feature_to_plot},
        yaxis={'title': '用户数'},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# 流失预测
elif analysis_type == "流失预测":
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    st.header("🔮 流失预测分析")

    # 预测性能卡片
    st.markdown("<div class='card animate-scale-in'>", unsafe_allow_html=True)
    st.subheader("📊 预测性能")
    st.markdown("<div style='display: flex; gap: 20px; flex-wrap: wrap;'>",
                unsafe_allow_html=True)
    st.markdown("<div style='background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin: 0;'>最佳模型</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.5rem; font-weight: bold;'>随机森林</p>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin: 0;'>模型准确率</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.5rem; font-weight: bold;'>94.02%</p>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 特征重要性
    st.markdown("<div class='card animate-slide-in'>", unsafe_allow_html=True)
    st.subheader("📋 特征重要性")
    feature_importance = pd.DataFrame({
        '特征': ['使用时长', '折扣金额', '仓库到家庭距离', '订单金额增长',
               '距上次订单天数', '关注主播数', '满意度', '订单数',
               '婚姻状况', '优惠券使用'],
        '重要性': [19.8, 14.3, 9.5, 8.2, 8.0, 7.6, 5.7, 4.4, 4.2, 4.0]
    })
    fig = px.bar(
        x=feature_importance['特征'],
        y=feature_importance['重要性'],
        color=feature_importance['重要性'],
        title="特征重要性",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(
        title={'text': "特征重要性", 'x': 0.5, 'xanchor': 'center'},
        xaxis={'title': '特征'},
        yaxis={'title': '重要性分数'},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    # 添加动画效果
    fig.update_traces(
        hoverinfo='x+y',
        marker=dict(
            opacity=0.8,
            line=dict(color='#ffffff', width=1)
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 流失概率分布
    st.markdown("<div class='card animate-fade-in'>", unsafe_allow_html=True)
    st.subheader("📈 流失概率分布")
    fig = px.histogram(
        df,
        x='ChurnProbability',
        nbins=50,
        title="流失概率分布",
        color_discrete_sequence=['#3498db']
    )
    fig.update_layout(
        title={'text': "流失概率分布", 'x': 0.5, 'xanchor': 'center'},
        xaxis={'title': '流失概率'},
        yaxis={'title': '用户数'},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    # 添加动画效果
    fig.update_traces(
        hoverinfo='x+y',
        marker=dict(
            opacity=0.8,
            line=dict(color='#ffffff', width=1)
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# 特征分析
elif analysis_type == "特征分析":
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    st.header("🔍 特征分析")

    selected_feature = st.selectbox(
        "选择特征",
        ['Tenure', 'OrderCount', 'HourSpendOnApp', 'SatisfactionScore',
         'DiscountAmount', 'DaySinceLastOrder']
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div class='card animate-slide-in' style='animation-delay: 0.1s;'>", unsafe_allow_html=True)
        st.subheader(f"{selected_feature} 分布")
        fig = px.histogram(
            df,
            x=selected_feature,
            color='Churn',
            barmode='overlay',
            title=f"{selected_feature} 分布",
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        fig.update_layout(
            title={'text': f"{selected_feature} 分布",
                   'x': 0.5, 'xanchor': 'center'},
            xaxis={'title': selected_feature},
            yaxis={'title': '用户数'},
            transition={'duration': 1000},
            template="plotly_white",
            font={'family': 'SimHei, Noto Sans SC, sans-serif'}
        )
        # 添加动画效果
        fig.update_traces(
            hoverinfo='x+y',
            marker=dict(
                opacity=0.7
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div class='card animate-slide-in' style='animation-delay: 0.2s;'>", unsafe_allow_html=True)
        st.subheader(f"{selected_feature} 与流失率关系")
        if selected_feature == 'SatisfactionScore':
            group_data = df.groupby(selected_feature)[
                'Churn'].mean().reset_index()
            group_data.columns = ['feature', 'Churn']
        else:
            # 将区间转换为字符串
            df['feature_bin'] = pd.cut(df[selected_feature], bins=5)
            group_data = df.groupby('feature_bin')[
                'Churn'].mean().reset_index()
            # 转换区间为字符串
            group_data['feature_bin'] = group_data['feature_bin'].astype(str)
            group_data.columns = ['feature', 'Churn']

        fig = px.bar(
            x=group_data['feature'],
            y=group_data['Churn']*100,
            title=f"{selected_feature} 与流失率关系",
            color_discrete_sequence=['#9b59b6']
        )
        fig.update_layout(
            title={'text': f"{selected_feature} 与流失率关系",
                   'x': 0.5, 'xanchor': 'center'},
            xaxis={'title': selected_feature},
            yaxis={'title': '流失率 (%)'},
            transition={'duration': 1000},
            template="plotly_white",
            font={'family': 'SimHei, Noto Sans SC, sans-serif'}
        )
        # 添加动画效果
        fig.update_traces(
            hoverinfo='x+y',
            marker=dict(
                opacity=0.8,
                line=dict(color='#ffffff', width=1)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# 交互式预测
elif analysis_type == "交互式预测":
    st.markdown("<div class='animate-fade-in'>", unsafe_allow_html=True)
    st.header("🔮 交互式流失预测")

    # 输入用户特征
    st.markdown("<div class='card animate-scale-in'>", unsafe_allow_html=True)
    st.subheader("📝 输入用户特征")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("使用时长 (月)", 1, 60, 12)
        city_tier = st.selectbox("城市等级", [1, 2, 3])
        warehouse_to_home = st.slider("仓库到家庭距离", 1, 50, 10)
        hour_spend_on_app = st.slider("APP使用时长 (小时)", 0.1, 10.0, 2.5)
        order_count = st.slider("订单数量", 1, 50, 3)
        order_amount_hike = st.slider("订单金额增长 (%)", 0, 100, 15)

    with col2:
        day_since_last_order = st.slider("距上次订单天数", 1, 30, 4)
        number_of_streamer_followed = st.slider("关注主播数", 0, 50, 5)
        satisfaction_score = st.selectbox("满意度评分", [1, 2, 3, 4, 5])
        coupon_used = st.slider("使用优惠券数量", 0, 20, 1)
        discount_amount = st.slider("折扣金额", 0, 500, 150)

        # 分类特征
        preferred_login_device = st.selectbox(
            "首选登录设备", ['Mobile Phone', 'Phone', 'Pad'])
        marital_status = st.selectbox(
            "婚姻状况", ['Married', 'Single', 'Divorced'])
        gender = st.selectbox("性别", ['Male', 'Female'])
        prefered_order_cat = st.selectbox("偏好订单类别",
                                          ['Laptop & Accessory', 'Mobile Phone', 'Fashion',
                                           'Household', 'Grocery', 'Others'])
    st.markdown("</div>", unsafe_allow_html=True)

    # 编码分类特征
    device_encoder = LabelEncoder()
    device_encoder.fit(['Mobile Phone', 'Phone', 'Pad'])
    device_encoded = device_encoder.transform([preferred_login_device])[0]

    marital_encoder = LabelEncoder()
    marital_encoder.fit(['Married', 'Single', 'Divorced'])
    marital_encoded = marital_encoder.transform([marital_status])[0]

    gender_encoder = LabelEncoder()
    gender_encoder.fit(['Male', 'Female'])
    gender_encoded = gender_encoder.transform([gender])[0]

    cat_encoder = LabelEncoder()
    cat_encoder.fit(['Laptop & Accessory', 'Mobile Phone', 'Fashion',
                    'Household', 'Grocery', 'Others'])
    cat_encoded = cat_encoder.transform([prefered_order_cat])[0]

    # 准备特征
    features = np.array([[tenure, city_tier, warehouse_to_home, hour_spend_on_app,
                         order_count, order_amount_hike, day_since_last_order,
                         number_of_streamer_followed, satisfaction_score, coupon_used,
                         discount_amount, device_encoded, marital_encoded,
                         gender_encoded, cat_encoded]])

    # 标准化
    features_scaled = scaler.transform(features)

    # 预测
    churn_prob = model.predict_proba(features_scaled)[0, 1]
    churn_pred = model.predict(features_scaled)[0]

    # 风险等级
    def risk_level(prob):
        if prob < 0.2:
            return '低风险'
        elif prob < 0.5:
            return '中风险'
        else:
            return '高风险'

    risk = risk_level(churn_prob)

    # 预测结果
    st.markdown("<div class='card animate-slide-in'>", unsafe_allow_html=True)
    st.subheader("📊 预测结果")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div style='background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 8px; text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin: 0;'>流失概率</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size: 1.8rem; font-weight: bold;'>{churn_prob*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        status_color = "#e74c3c" if churn_pred == 1 else "#27ae60"
        status_text = "流失" if churn_pred == 1 else "留存"
        st.markdown(
            f"<div style='background: linear-gradient(135deg, {status_color}, {status_color}80); color: white; padding: 20px; border-radius: 8px; text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin: 0;'>预测结果</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size: 1.8rem; font-weight: bold;'>{status_text}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        risk_color = "#e74c3c" if risk == '高风险' else "#f39c12" if risk == '中风险' else "#27ae60"
        st.markdown(
            f"<div style='background: linear-gradient(135deg, {risk_color}, {risk_color}80); color: white; padding: 20px; border-radius: 8px; text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin: 0;'>风险等级</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size: 1.8rem; font-weight: bold;'>{risk}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 建议
    st.markdown("<div class='card animate-fade-in'>", unsafe_allow_html=True)
    st.subheader("💡 营销建议")
    if risk == '高风险':
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top: 0;'>该用户流失风险较高，建议：</h4>",
                    unsafe_allow_html=True)
        st.markdown("• 🎁 发送个性化优惠券")
        st.markdown("• 📞 提供专属客服服务")
        st.markdown("• 🛍️ 推送个性化产品推荐")
        st.markdown("</div>", unsafe_allow_html=True)
    elif risk == '中风险':
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top: 0;'>该用户存在一定流失风险，建议：</h4>",
                    unsafe_allow_html=True)
        st.markdown("• 📅 定期发送活动通知")
        st.markdown("• 🎉 提供会员专属福利")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top: 0;'>该用户流失风险较低，建议：</h4>",
                    unsafe_allow_html=True)
        st.markdown("• 👥 维护现有关系")
        st.markdown("• 📤 鼓励用户分享推荐")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 预测概率可视化
    st.markdown("<div class='card animate-scale-in'>", unsafe_allow_html=True)
    st.subheader("📈 预测概率分析")
    fig = px.pie(
        names=['留存概率', '流失概率'],
        values=[1-churn_prob, churn_prob],
        color=['#27ae60', '#e74c3c'],
        title="预测概率分布",
        hole=0.5
    )
    fig.update_layout(
        title={'text': "预测概率分布", 'x': 0.5, 'xanchor': 'center'},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.1},
        transition={'duration': 1000},
        template="plotly_white",
        font={'family': 'SimHei, Noto Sans SC, sans-serif'}
    )
    # 添加动画效果
    fig.update_traces(
        hoverinfo='label+percent+value',
        textinfo='label+percent',
        textfont_size=14,
        marker=dict(
            line=dict(color='#ffffff', width=2)
        ),
        opacity=0.8
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("© 2026 电商用户分析系统 | 数据更新时间: 2026-03-04")
