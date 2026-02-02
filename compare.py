import pandas as pd

# =========================
# 1) 读取三个结果表
# =========================
df_base = pd.read_csv("season11_map_vhat_baseline.csv")
df_m5   = pd.read_csv("season11_map_vhat_minus5.csv")
df_p5   = pd.read_csv("season11_map_vhat_plus5.csv")


# =========================
# 2) 提取“每周淘汰人物”
# =========================
def extract_eliminated(df, label):
    """
    返回 DataFrame:
    week | eliminated_<label>
    """
    out = (
        df[df["is_eliminated_that_week"] == 1]
        [["week", "celebrity_name"]]
        .rename(columns={"celebrity_name": f"eliminated_{label}"})
        .sort_values("week")
        .reset_index(drop=True)
    )
    return out


elim_base = extract_eliminated(df_base, "baseline")
elim_m5   = extract_eliminated(df_m5, "minus5")
elim_p5   = extract_eliminated(df_p5, "plus5")


# =========================
# 3) 合并三种情景
# =========================
df_compare = (
    elim_base
    .merge(elim_m5, on="week", how="inner")
    .merge(elim_p5, on="week", how="inner")
)


# =========================
# 4) 是否与 baseline 一致
# =========================
df_compare["same_as_baseline"] = (
    (df_compare["eliminated_baseline"] == df_compare["eliminated_minus5"]) &
    (df_compare["eliminated_baseline"] == df_compare["eliminated_plus5"])
)


# =========================
# 5) 输出结果
# =========================
print(df_compare)
df_compare.to_csv("season11_elimination_sensitivity_comparison.csv", index=False)
print("Saved: season11_elimination_sensitivity_comparison.csv")
