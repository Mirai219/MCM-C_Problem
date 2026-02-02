import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# =========================
# 1) 文件路径
# =========================
data_file_path = '2026_MCM_Problem_C_Data.csv'
season11_file_path = 'season11_weekly_detail_table.csv'

# =========================
# 2) 读取数据
# =========================
df_data = pd.read_csv(data_file_path)
df_season11 = pd.read_csv(season11_file_path)

df_age = df_data[['celebrity_name', 'celebrity_age_during_season']]
df_scores = df_season11[['celebrity_name', 'rank_sum_rank']]

df_merged = pd.merge(df_age, df_scores, on='celebrity_name', how='inner')

base_age = df_merged['celebrity_age_during_season'].to_numpy(dtype=float)
rank_sum = df_merged['rank_sum_rank'].to_numpy(dtype=float)

# =========================
# 3) Monte Carlo 敏感度分析
# =========================
np.random.seed(42)      # 保证可复现
N_SIM = 1000            # 模拟次数
PERTURB = 0.05          # ±5%

corrs = []

for _ in range(N_SIM):
    noise = np.random.uniform(-PERTURB, PERTURB, size=base_age.shape)
    age_perturbed = base_age * (1.0 + noise)

    corr, _ = spearmanr(age_perturbed, rank_sum)
    corrs.append(corr)

corrs = np.array(corrs)

# =========================
# 4) 基准值
# =========================
base_corr, _ = spearmanr(base_age, rank_sum)

# =========================
# 5) 输出统计结果
# =========================
print(f"Baseline Spearman Correlation: {base_corr:.6f}")
print(f"Monte Carlo mean correlation: {corrs.mean():.6f}")
print(f"Monte Carlo std  correlation: {corrs.std():.6f}")
print(f"Min / Max correlation: {corrs.min():.6f} / {corrs.max():.6f}")

# =========================
# 6) 保存结果
# =========================
df_out = pd.DataFrame({
    "simulation_id": np.arange(N_SIM),
    "spearman_correlation": corrs
})

df_out.to_csv("age_random_sensitivity_spearman.csv", index=False)
print("Saved: age_random_sensitivity_spearman.csv")
