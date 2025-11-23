# 营销数据分析师
print("开始客户聚类分析...")
print("步骤1: 导入必要的库")

try:
    # 1. 导入所有必要的库
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import warnings
    warnings.filterwarnings('ignore')
    print("✓ 所有库导入成功！")

except ImportError as e:
    print(f"❌ 导入库时出错: {e}")
    print("请确保已安装所有必需的库。在终端中运行以下命令：")
    print("pip install pandas numpy scikit-learn matplotlib seaborn")
    exit()

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("\n步骤2: 加载您的数据")

csv_file_path = "E:\\大二上材料\\市场营销学原理\\作业四\\used_data.csv"  # 请将 'your_data.csv' 替换为您的实际文件名
try:
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    print(f"✓ 数据文件加载成功: {csv_file_path}")
    
except FileNotFoundError:
    print(f"❌ 找不到文件: {csv_file_path}")
    print("请检查：")
    print("1. 文件路径是否正确")
    print("2. 文件名是否拼写正确（包括.csv后缀）")
    print("3. 文件是否在指定目录中")
    exit()
except Exception as e:
    print(f"❌ 读取文件时出错: {e}")
    exit()

print(f"数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")

print("\n步骤3: 检查数据结构和列名")
print("数据前5行:")
print(df.head())

print("\n数据列名:")
print(df.columns.tolist())

print("\n数据基本信息:")
print(df.info())

#CSV有表头且列名就是以下名称，
expected_columns = {
    'customer_id': df.columns[0],  # 第一列：顾客id
    'total_spent': df.columns[1],   # 第二列：总金额数
    'num_orders': df.columns[2],    # 第三列：订单数量
    'avg_order_value': df.columns[3],  # 第四列：平均订单价值
    'Electronics_amount': df.columns[4]  # 第五列：电子产品消费
}

print("检测到的列名映射:")
for key, value in expected_columns.items():
    print(f"  {key}: {value}")

# 重命名列以便代码统一处理
df_clean = df.rename(columns={
    expected_columns['customer_id']: 'customer_id',
    expected_columns['total_spent']: 'total_spent',
    expected_columns['num_orders']: 'num_orders',
    expected_columns['avg_order_value']: 'avg_order_value',
    expected_columns['Electronics_amount']: 'Electronics_amount'
})

print("\n步骤4: 数据质量检查")
print("数据基本信息:")
print(df_clean.info())

print("\n描述性统计:")
print(df_clean[['total_spent', 'num_orders', 'avg_order_value', 'Electronics_amount']].describe())

print("\n缺失值检查:")
missing_values = df_clean[['total_spent', 'num_orders', 'avg_order_value', 'Electronics_amount']].isnull().sum()
print(missing_values)

# 处理缺失值（如果有）
if missing_values.sum() > 0:
    print("发现缺失值，使用中位数填充...")
    df_clean = df_clean.fillna(df_clean[['total_spent', 'num_orders', 'avg_order_value', 'Electronics_amount']].median())
    print("✓ 缺失值处理完成")

print("\n步骤5: 选择聚类变量")
# 选择四个关键变量进行聚类
cluster_vars = ['total_spent', 'num_orders', 'avg_order_value', 'Electronics_amount']
cluster_data = df_clean[cluster_vars]

print("✓ 选择的聚类变量:")
for i, var in enumerate(cluster_vars, 1):
    print(f"  {i}. {var}")

print("\n步骤6: 数据标准化")
# 数据标准化 - 这是非常重要的一步！
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# 转换为DataFrame便于查看
cluster_data_scaled_df = pd.DataFrame(cluster_data_scaled, columns=cluster_vars)
print("✓ 数据标准化完成")
print("\n标准化后的数据统计:")
print(cluster_data_scaled_df.describe())

print("\n步骤7: 寻找最优K值")
print("正在计算不同K值下的聚类效果...")

# 测试不同的K值（从2到10）
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    # 创建KMeans模型
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_data_scaled)
    
    # 记录惯性（簇内平方和）
    inertias.append(kmeans.inertia_)
    
    # 计算轮廓系数
    labels = kmeans.labels_
    score = silhouette_score(cluster_data_scaled, labels)
    silhouette_scores.append(score)
    
    print(f"K={k}: 惯性 = {kmeans.inertia_:.2f}, 轮廓系数 = {score:.4f}")

print("\n步骤8: 可视化K值选择结果")
# 创建可视化图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 肘部法则图
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('聚类数量 (K)')
ax1.set_ylabel('惯性 (Inertia)')
ax1.set_title('肘部法则 - 寻找最优K值')
ax1.grid(True, alpha=0.3)

# 在肘部图上标记可能的拐点
elbow_k = 4  # 根据图形判断，可以调整
ax1.axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, label=f'可能拐点 K={elbow_k}')
ax1.legend()

# 轮廓系数图
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('聚类数量 (K)')
ax2.set_ylabel('轮廓系数')
ax2.set_title('轮廓系数 - 寻找最优K值')
ax2.grid(True, alpha=0.3)

# 标记最佳轮廓系数
best_k_index = np.argmax(silhouette_scores)
best_k = k_range[best_k_index]
best_score = silhouette_scores[best_k_index]

ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, 
            label=f'最佳K值: {best_k} (分数: {best_score:.3f})')
ax2.legend()

plt.tight_layout()
plt.savefig('k_value_selection.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ 图表已保存为 'k_value_selection.png'")
print(f"根据轮廓系数，建议的最佳K值是: {best_k}")
print(f"对应的轮廓系数: {best_score:.4f}")

print("\n步骤9: 执行最终聚类")
# 使用最佳K值进行最终聚类
final_k = best_k  # 您也可以根据图表手动选择，比如选择 elbow_k

print(f"使用 K={final_k} 进行最终聚类...")

final_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
final_kmeans.fit(cluster_data_scaled)

# 将聚类结果添加到原始数据
df_clean['cluster'] = final_kmeans.labels_

print("✓ 聚类完成")
print("\n各簇客户分布:")
cluster_counts = df_clean['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(df_clean)) * 100
    print(f"簇 {cluster_id}: {count} 位客户 ({percentage:.1f}%)")

print("\n步骤10: 分析聚类结果")
# 分析每个簇的特征
print("\n各簇在关键变量上的平均值:")
cluster_profile = df_clean.groupby('cluster')[cluster_vars].mean()
print(cluster_profile.round(2))

# 计算与总体均值的相对差异
print("\n各簇与总体均值的相对差异 (%):")
relative_diff = (cluster_profile / cluster_data.mean() - 1) * 100
print(relative_diff.round(2))

print("\n步骤11: 可视化聚类特征")
# 创建簇特征可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700', '#FFB6C1']

for i, var in enumerate(cluster_vars):
    # 为每个变量创建箱线图
    box_data = [df_clean[df_clean['cluster'] == cluster][var] for cluster in range(final_k)]
    axes[i].boxplot(box_data, labels=range(final_k), patch_artist=True)
    axes[i].set_title(f'{var} 的分布 by 簇')
    axes[i].set_ylabel(var)
    axes[i].set_xlabel('簇')
    
    # 添加颜色
    for patch, color in zip(axes[i].artists, colors[:final_k]):
        patch.set_facecolor(color)

plt.suptitle('各变量在不同簇中的分布', fontsize=16)
plt.tight_layout()
plt.savefig('cluster_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n步骤12: 创建综合特征对比图")
plt.figure(figsize=(14, 10))

# 准备数据用于条形图
melted_data = df_clean.melt(id_vars=['cluster'], value_vars=cluster_vars, 
                      var_name='指标', value_name='值')

# 创建分组条形图
plt.subplot(2, 1, 1)
sns.barplot(data=melted_data, x='cluster', y='值', hue='指标', palette='viridis')
plt.title('各客户簇的特征对比', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 创建客户分布饼图
plt.subplot(2, 1, 2)
colors = plt.cm.Set3(np.linspace(0, 1, final_k))
plt.pie(cluster_counts.values, labels=[f'簇 {i}' for i in cluster_counts.index], 
        autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('客户簇分布比例', fontsize=14)

plt.tight_layout()
plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n步骤13: 生成详细的业务解读报告")
print("=" * 60)
print("           客户聚类分析业务报告")
print("=" * 60)

print(f"\n📊 分析总结:")
print(f"   • 总客户数: {len(df_clean)}")
print(f"   • 最优聚类数: {final_k}")
print(f"   • 聚类质量 (轮廓系数): {best_score:.3f}")
print(f"   • 轮廓系数解读: {'优秀' if best_score > 0.7 else '良好' if best_score > 0.5 else '一般'}")

print(f"\n👥 各客户群详细描述:")

# 为每个簇生成详细描述
for cluster_id in range(final_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]
    cluster_mean = cluster_data[cluster_vars].mean()
    total_mean = df_clean[cluster_vars].mean()
    
    print(f"\n🎯 客户群 {cluster_id} (占比: {len(cluster_data)/len(df_clean)*100:.1f}%)")
    print(f"   📈 关键指标:")
    print(f"      • 总消费: ¥{cluster_mean['total_spent']:.0f} "
          f"({'+' if cluster_mean['total_spent'] > total_mean['total_spent'] else ''}"
          f"{(cluster_mean['total_spent']/total_mean['total_spent']-1)*100:.0f}%)")
    print(f"      • 订单数: {cluster_mean['num_orders']:.1f} "
          f"({'+' if cluster_mean['num_orders'] > total_mean['num_orders'] else ''}"
          f"{(cluster_mean['num_orders']/total_mean['num_orders']-1)*100:.0f}%)")
    print(f"      • 客单价: ¥{cluster_mean['avg_order_value']:.0f} "
          f"({'+' if cluster_mean['avg_order_value'] > total_mean['avg_order_value'] else ''}"
          f"{(cluster_mean['avg_order_value']/total_mean['avg_order_value']-1)*100:.0f}%)")
    print(f"      • 电子产品消费: ¥{cluster_mean['Electronics_amount']:.0f} "
          f"({'+' if cluster_mean['Electronics_amount'] > total_mean['Electronics_amount'] else ''}"
          f"{(cluster_mean['Electronics_amount']/total_mean['Electronics_amount']-1)*100:.0f}%)")
   
# 保存结果
output_file = 'customer_clustering_results.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"结果已保存到: {output_file}")
print(f"图表已保存为: k_value_selection.png, cluster_distributions.png, cluster_comparison.png")


# 显示完成消息
print("客户聚类分析已完成")
