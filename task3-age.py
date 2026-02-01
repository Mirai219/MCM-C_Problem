import pandas as pd
from scipy.stats import spearmanr

# 加载数据
data_file_path = '2026_MCM_Problem_C_Data.csv'
season11_file_path = 'season11_weekly_detail_table.csv'

# 加载数据
df_data = pd.read_csv(data_file_path)
df_season11 = pd.read_csv(season11_file_path)

# 提取年龄和成绩（rank_sum_rank）
df_age = df_data[['celebrity_name', 'celebrity_age_during_season']]
df_scores = df_season11[['celebrity_name', 'rank_sum_rank']]

# 合并数据：根据姓名进行合并
df_merged = pd.merge(df_age, df_scores, on='celebrity_name', how='inner')

# 计算Spearman相关系数
age = df_merged['celebrity_age_during_season']
rank_sum = df_merged['rank_sum_rank']

# 计算Spearman相关系数
corr, _ = spearmanr(age, rank_sum)

# 输出相关系数
print(f"Spearman Correlation between Age and Rank Sum: {corr}")
