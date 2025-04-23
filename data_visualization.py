import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/Mike/Desktop/Results_21Mar2022.csv')




'''
 - 平行坐标图
# 构造 CEIS（等权重）
ceis_columns = [
    'mean_ghgs', 'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_land',
    'mean_watscar', 'mean_watuse', 'mean_eut', 'mean_acid', 'mean_bio'
]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[ceis_columns])
df['CEIS'] = normalized_data.mean(axis=1)

# 创建平行坐标图数据（只取部分代表字段以避免维度太多）
selected_cols = ['CEIS'] + ceis_columns[:6] + ['diet_group']  # 取6个环境维度+CEIS+diet_group
plot_df = df[selected_cols].copy()

# 再次归一化用于平行坐标图（确保视觉一致性）
norm_cols = selected_cols[:-1]  # 除了 diet_group 之外的所有列
plot_df[norm_cols] = MinMaxScaler().fit_transform(plot_df[norm_cols])

plot_df = df[selected_cols].copy()
plot_df[selected_cols[:-1]] = MinMaxScaler().fit_transform(plot_df[selected_cols[:-1]])

# 对 diet_group 进行编码，用于分类着色
le = LabelEncoder()
plot_df['diet_group_encoded'] = le.fit_transform(plot_df['diet_group'])

custom_colors = ['#a50026', '#d73027', '#f46d43', '#74add1', '#4575b4', '#313695']  # 6色红到蓝

# 使用 diet_group 分类着色生成平行坐标图
fig = px.parallel_coordinates(
    plot_df,
    color='diet_group_encoded',
    dimensions=selected_cols[:-1],
    labels={col: col for col in selected_cols[:-1]},
    color_continuous_scale=custom_colors, #px.colors.qualitative.Set1,
    title="Parallel Coordinates Plot: Environmental Profiles Colored by Diet Type"
)

fig.update_coloraxes(colorbar_title='diet_group', colorbar_tickvals=list(range(len(le.classes_))),
                     colorbar_ticktext=list(le.classes_))
fig.show()
'''


'''
# 构造 CEIS（等权重）
ceis_columns = [
    'mean_ghgs', 'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_land',
    'mean_watscar', 'mean_watuse', 'mean_eut', 'mean_acid', 'mean_bio'
]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[ceis_columns])
df['CEIS'] = normalized_data.mean(axis=1)

# 创建平行坐标图数据（只取部分代表字段以避免维度太多）
selected_cols = ['CEIS'] + ceis_columns[:6] + ['diet_group']  # 取6个环境维度+CEIS+diet_group
plot_df = df[selected_cols].copy()

# 再次归一化用于平行坐标图（确保视觉一致性）
norm_cols = selected_cols[:-1]  # 除了 diet_group 之外的所有列
plot_df[norm_cols] = MinMaxScaler().fit_transform(plot_df[norm_cols])

# 生成交互式平行坐标图
fig = px.parallel_coordinates(
    plot_df,
    color='CEIS',
    dimensions=norm_cols,
    color_continuous_scale='RdBu_r',
    labels={col: col for col in norm_cols},
    title="Parallel Coordinates Plot: Environmental Impact Profiles by Diet Type"
)

fig.show()
'''

# 定义变量分组
ghg_related = ['mean_ghgs', 'mean_ghgs_ch4', 'mean_ghgs_n2o']
land_water = ['mean_land', 'mean_watuse']
eco_harm = ['mean_bio', 'mean_acid', 'mean_eut', 'mean_watscar']
ehi_columns = ghg_related + land_water + eco_harm

# 标准化指标
scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(df[ehi_columns]), columns=ehi_columns)

# 权重设定
weights = {
    **{col: 0.2 / len(ghg_related) for col in ghg_related},
    **{col: 0.2 / len(land_water) for col in land_water},
    **{col: 0.6 / len(eco_harm) for col in eco_harm},
}

weights_2 = {    
    **{col: 0.6 / len(ghg_related) for col in ghg_related},
    **{col: 0.2 / len(land_water) for col in land_water},
    **{col: 0.2 / len(eco_harm) for col in eco_harm},
}

# 计算 EHI（综合环境影响指数）
df['EHI_eco_harm'] = sum(normalized[col] * weights[col] for col in ehi_columns)
df['EHI_green_house_harm'] = sum(normalized[col] * weights_2[col] for col in ehi_columns)

# 构建 Treemap
fig = px.treemap(
    df,
    path=['diet_group', 'sex', 'age_group'],
    values='EHI_green_house_harm',       # 面积 = 总环境负荷
    color='EHI_eco_harm',              # 颜色 = 人均环境影响
    color_continuous_scale='RdBu_r',
    title='Treemap: Total Environmental Harm by Group (Area = EHI_greenhouse_harm, Color = EHI_eco_harm)',
)

fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig.show()

'''
#2 - Treemap - 饮食类型、性别和年龄组的温室气体排放
# Create Treemap based on diet_group → sex → age_group hierarchy
treemap_data = df.copy()
#treemap_data['eut_impact_total'] =  df['n_participants'] * df['mean_eut']
# 计算总 GHG 排放量 = n_participants × mean_ghgs
df['eut_impact_total'] = df['n_participants'] * df['mean_eut']

# 按 diet_group 聚合：总排放与总人数
aggregated = df.groupby('diet_group').agg(
    eut_impact_total=('eut_impact_total', 'sum'),
    total_participants=('n_participants', 'sum')
).reset_index()

# 计算加权平均 GHG 排放
aggregated['weighted_avg_eut'] = aggregated['eut_impact_total'] / aggregated['total_participants']

treemap_data = treemap_data.merge(
    aggregated[['diet_group', 'weighted_avg_eut']],
    on='diet_group',
    how='left'
)

# Use mean_ghgs as area size
fig = px.treemap(
    treemap_data,
    path=['diet_group', 'sex', 'age_group'],
    values='weighted_avg_eut',
    color='mean_eut',
    color_continuous_scale='RdBu_r',
    title='Treemap: Eutrophication Potential by Diet Type, Gender, and Age Group'
)

fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig.show()
'''


'''
2 - 雷达图 - 不同饮食类型的环境影响
# chose the columns to be used for clustering
env_columns = [
    'mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut',
    'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_bio',
    'mean_watuse', 'mean_acid'
]

# base on the grouping column, we can get the mean of each group
radar_data = df.groupby('diet_group')[env_columns].mean()
print(radar_data.head())

# standardize the data for radar chart
radar_data_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

# 1 parameter each row, 1 diet each column
radar_data_transposed = radar_data_normalized.T


# create radar chart
fig = go.Figure()

for diet in radar_data_transposed.columns:
    fig.add_trace(go.Scatterpolar(
        r=radar_data_transposed[diet].values,
        theta=radar_data_transposed.index,
        fill='toself',
        name=diet.capitalize()
    ))

# picture config
fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    title="Radar Chart: Environmental Impacts by Diet Type (Normalized)"
)

fig.show()
'''

'''
3 - 层级聚类热图
聚类分析

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[env_columns])

# create a new DataFrame with the scaled data and the grouping column
df_scaled = pd.DataFrame(X_scaled, index=df['grouping'], columns=env_columns)

plt.figure(figsize=(12, 10))

# generate a heatmap with dendrograms
sns.set(font_scale=0.8)
clustergrid = sns.clustermap(
    df_scaled,
    method='ward',
    metric='euclidean',
    cmap='vlag',
    figsize=(14, 12),
    dendrogram_ratio=(.1, .2),
    cbar_pos=(0.02, 0.8, 0.02, 0.18),
    yticklabels=True
)

plt.title("Hierarchical Clustered Heatmap of Environmental Impacts by Diet Group", pad=100)
plt.show()
'''