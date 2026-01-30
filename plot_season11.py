import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "season11_map_vhat.csv"   # 改成你的路径
TOP_K = 5                            # Top-k 折线图默认画 5 个


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 基础清洗
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype(int)
    df["v_hat"] = pd.to_numeric(df["v_hat"], errors="coerce")
    df["is_eliminated_that_week"] = pd.to_numeric(df["is_eliminated_that_week"], errors="coerce").fillna(0).astype(int)
    return df


def plot_heatmap(df: pd.DataFrame, season: int = 11, sort_by: str = "mean"):
    """
    热力图：行=选手，列=周，值=v_hat
    sort_by:
      - "mean": 按全季平均支持度排序（推荐）
      - "last": 按最后出现那周的支持度排序
    """
    d = df[df["season"] == season].copy()
    if d.empty:
        raise ValueError(f"No data for season={season}")

    pivot = d.pivot_table(index="celebrity_name", columns="week", values="v_hat", aggfunc="mean")

    # 排序（让趋势更清楚）
    if sort_by == "mean":
        order = pivot.mean(axis=1, skipna=True).sort_values(ascending=False).index
        pivot = pivot.loc[order]
    elif sort_by == "last":
        # 每个选手最后出现的那周 v_hat
        last_val = pivot.apply(lambda row: row.dropna().iloc[-1] if row.dropna().size > 0 else np.nan, axis=1)
        order = last_val.sort_values(ascending=False).index
        pivot = pivot.loc[order]

    # 处理 NaN：淘汰后该选手不再出现，用 mask 让热力图留白
    mat = pivot.to_numpy()
    mat_masked = np.ma.masked_invalid(mat)

    plt.figure(figsize=(12, max(6, 0.28 * pivot.shape[0])))
    im = plt.imshow(mat_masked, aspect="auto")
    plt.colorbar(im, label="v_hat (estimated fan support share)")

    plt.title(f"Season {season}: Fan Support (v_hat) Heatmap")
    plt.xlabel("Week")
    plt.ylabel("Contestant")

    # x 轴周标签
    weeks = pivot.columns.tolist()
    plt.xticks(ticks=np.arange(len(weeks)), labels=weeks)

    # y 轴选手标签（人多时字体调小）
    fontsize = 10 if pivot.shape[0] <= 25 else 7
    plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index.tolist(), fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"season{season}_heatmap_vhat.png", dpi=200)
    plt.show()


def plot_topk_lines(df: pd.DataFrame, season: int = 11, top_k: int = 5):
    """
    折线图：选整季平均 v_hat 最高的 Top-k 选手，画其随周变化曲线；
    并在淘汰周打 x 标记。
    """
    d = df[df["season"] == season].copy()
    if d.empty:
        raise ValueError(f"No data for season={season}")

    # 选 Top-k：按全季平均 v_hat 排序
    mean_support = d.groupby("celebrity_name")["v_hat"].mean().sort_values(ascending=False)
    top_names = mean_support.head(top_k).index.tolist()

    plt.figure(figsize=(12, 6))
    for name in top_names:
        sub = d[d["celebrity_name"] == name].sort_values("week")
        plt.plot(sub["week"], sub["v_hat"], marker="o", linewidth=2, label=name)

        # 在淘汰周标记（如果该选手在本季被淘汰且记录到了淘汰周）
        elim = sub[sub["is_eliminated_that_week"] == 1]
        if not elim.empty:
            ew = int(elim["week"].iloc[0])
            ev = float(elim["v_hat"].iloc[0])
            plt.scatter([ew], [ev], marker="x", s=120)

    weeks = sorted(d["week"].unique().tolist())
    plt.xticks(weeks)
    plt.xlabel("Week")
    plt.ylabel("v_hat (estimated fan support share)")
    plt.title(f"Season {season}: Top-{top_k} Contestants Support Trend (MAP v_hat)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"season{season}_top{top_k}_trend_vhat.png", dpi=200)
    plt.show()


def main():
    df = load_data(CSV_PATH)

    # 1) 全景热力图
    plot_heatmap(df, season=11, sort_by="mean")

    # 2) Top-k 折线趋势
    plot_topk_lines(df, season=11, top_k=TOP_K)


if __name__ == "__main__":
    main()
