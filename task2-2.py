import pandas as pd
from scipy.stats import spearmanr

# 读取数据文件
file_path = "season11_weekly_detail_table.csv"
df = pd.read_csv(file_path)

# 选择要分析的列
columns_to_analyze = ['judge_rank', 'fan_rank', 'score_pct_rank', 'rank_sum_rank']

# 计算Spearman相关系数
spearman_results = {}

# 计算每一对之间的Spearman相关系数
for i in range(len(columns_to_analyze)):
    for j in range(i+1, len(columns_to_analyze)):
        col1 = columns_to_analyze[i]
        col2 = columns_to_analyze[j]
        corr, _ = spearmanr(df[col1], df[col2])
        spearman_results[f'{col1} vs {col2}'] = corr

# 输出结果
print("Spearman Correlation Results:")
for key, value in spearman_results.items():
    print(f"{key}: {value}")
