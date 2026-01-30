import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# =========================
# 1) 数据结构
# =========================
@dataclass
class WeekData:
    season: int
    week: int
    alive_ids: List[str]          # 当周仍在赛的选手ID（这里用 celebrity_name 作为ID）
    judge_share: np.ndarray       # shape (n_alive,)
    eliminated_idx: int           # 淘汰者在 alive_ids 中的索引


# =========================
# 2) 解析列名：weekX_judgeY_score
# =========================
_WEEK_JUDGE_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")

def find_week_judge_columns(df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    返回：{week_num: [colnames...]}，每周对应那几位 judge 的分数列
    """
    week_cols: Dict[int, List[str]] = {}
    for c in df.columns:
        m = _WEEK_JUDGE_RE.match(c)
        if m:
            w = int(m.group(1))
            week_cols.setdefault(w, []).append(c)

    # 按 judge 序号排序（可选）
    for w in week_cols:
        week_cols[w] = sorted(
            week_cols[w],
            key=lambda x: int(_WEEK_JUDGE_RE.match(x).group(2))
        )
    return dict(sorted(week_cols.items(), key=lambda kv: kv[0]))


def compute_judge_totals_by_week(season_df: pd.DataFrame, week_cols: Dict[int, List[str]]) -> pd.DataFrame:
    """
    计算每个选手每周的裁判总分 J_{i,t}，忽略 N/A（读入后会变 NaN）
    返回：一个新 DataFrame，包含列：['celebrity_name', 'season', 'J_week1', 'J_week2', ...]
    """
    out = season_df[['celebrity_name', 'season']].copy()

    for w, cols in week_cols.items():
        # 转成数值，N/A -> NaN
        numeric = season_df[cols].apply(pd.to_numeric, errors='coerce')
        # 裁判总分：按行求和，跳过 NaN
        out[f"J_week{w}"] = numeric.sum(axis=1, skipna=True)

        # 注意：如果该周整周不存在，所有列可能都是 NaN -> sum 会得到 0.0
        # 但题目数据“赛季不存在的周”是 N/A；sum 变 0 可能混淆。
        # 我们用“该周所有judge列是否全NaN”来识别“周不存在”，并设为 NaN。
        all_nan_mask = numeric.isna().all(axis=1)
        out.loc[all_nan_mask, f"J_week{w}"] = np.nan

    return out


# =========================
# 3) 构造 WeekData（只保留“恰好淘汰1人”的周）
# =========================
def build_week_data_for_season_11(df: pd.DataFrame) -> List[WeekData]:
    """
    只取 season=11，构造可用于建模的 WeekData 列表。
    过滤规则：每周恰好淘汰1人（本周>0 且下周==0 的人恰好1个）。
    """
    season = 11
    season_df = df[df['season'] == season].copy()
    if season_df.empty:
        raise ValueError("season=11 数据为空，请检查 CSV 是否正确读取，以及 season 列是否存在/为整数。")

    week_cols = find_week_judge_columns(season_df)
    if not week_cols:
        raise ValueError("未找到 weekX_judgeY_score 列，请检查 CSV 表头是否与题目一致。")

    jt = compute_judge_totals_by_week(season_df, week_cols)

    # 找到该赛季最大周数（按列推断）
    weeks = sorted([int(re.search(r"J_week(\d+)", c).group(1)) for c in jt.columns if c.startswith("J_week")])
    max_week = max(weeks)

    week_data_list: List[WeekData] = []

    for w in range(1, max_week):  # 用 w 与 w+1 判断淘汰，所以到 max_week-1
        cur_col = f"J_week{w}"
        nxt_col = f"J_week{w+1}"
        if cur_col not in jt.columns or nxt_col not in jt.columns:
            continue

        # 当周存在且仍在赛：J>0（且不是 NaN）
        cur = jt[cur_col]
        nxt = jt[nxt_col]

        alive_mask = (cur.notna()) & (cur > 0)
        alive_df = jt.loc[alive_mask, ['celebrity_name', cur_col, nxt_col]].copy()
        if alive_df.empty:
            continue

        # 淘汰判定：本周>0，且下周==0（注意下周 NaN 表示周不存在/未播出，不当作淘汰）
        eliminated_mask = (alive_df[nxt_col].notna()) & (alive_df[nxt_col] == 0)
        eliminated_candidates = alive_df.loc[eliminated_mask, 'celebrity_name'].tolist()

        # 只保留“恰好淘汰1人”的周（符合你的简化假设）
        if len(eliminated_candidates) != 1:
            continue

        eliminated_name = eliminated_candidates[0]
        alive_ids = alive_df['celebrity_name'].tolist()

        # 计算 judge_share（当周裁判占比）
        judge_totals = alive_df[cur_col].to_numpy(dtype=float)
        denom = judge_totals.sum()
        if denom <= 0:
            continue
        judge_share = judge_totals / denom

        eliminated_idx = alive_ids.index(eliminated_name)

        week_data_list.append(
            WeekData(
                season=season,
                week=w,
                alive_ids=alive_ids,
                judge_share=judge_share,
                eliminated_idx=eliminated_idx
            )
        )

    if not week_data_list:
        raise ValueError("没有构造出任何可用周（可能该季存在多淘汰/无淘汰周，或淘汰识别规则需要调整）。")

    return week_data_list


# =========================
# 4) 单周 MAP 求解（PyTorch + LBFGS）
# =========================
def solve_week_map(
    week_data: WeekData,
    w: float = 0.5,
    beta: float = 10.0,
    alpha: float = 1.0,
    max_iter: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """
    返回：v_hat，shape (n_alive,)
    - v = softmax(z)
    - S = w*judge_share + (1-w)*v
    - p(elim=i) = softmax(-beta*S)
    - prior: Dirichlet(alpha) -> sum((alpha-1)*log(v))
    - MAP: maximize loglik + logprior  <=> minimize loss = -(loglik + logprior)
    """
    torch.manual_seed(seed)

    judge_share = torch.tensor(week_data.judge_share, dtype=torch.double)
    n = judge_share.numel()
    elim = int(week_data.eliminated_idx)

    # 无约束参数 z
    z = torch.zeros(n, dtype=torch.double, requires_grad=True)

    # LBFGS 优化器
    optimizer = torch.optim.LBFGS([z], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    eps = 1e-12
    alpha_tensor = torch.full((n,), float(alpha), dtype=torch.double)

    def closure():
        optimizer.zero_grad()

        v = torch.softmax(z, dim=0)  # (n,)
        S = float(w) * judge_share + (1.0 - float(w)) * v

        # log-likelihood: log softmax(-beta*S)[elim]
        logits = -float(beta) * S
        log_probs = torch.log_softmax(logits, dim=0)
        loglik = log_probs[elim]

        # log-prior (Dirichlet): sum((alpha-1)*log(v))
        logprior = torch.sum((alpha_tensor - 1.0) * torch.log(v + eps))

        loss = -(loglik + logprior)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        v_hat = torch.softmax(z, dim=0).cpu().numpy()
    return v_hat


# =========================
# 5) 全流程：读取 -> 取第11季 -> 构造周数据 -> 逐周MAP -> 输出表
# =========================
def run_season_11_map(
    csv_path: str,
    w: float = 0.5,
    beta: float = 10.0,
    alpha: float = 1.0,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 基础列检查
    required_cols = {"season", "celebrity_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}。请检查表头。")

    # season 列转 int（防止读成字符串）
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df = df[df["season"].notna()].copy()
    df["season"] = df["season"].astype(int)

    # 构造 WeekData 列表（只含 season=11，且每周恰好淘汰1人）
    week_data_list = build_week_data_for_season_11(df)

    records = []
    for wd in week_data_list:
        v_hat = solve_week_map(wd, w=w, beta=beta, alpha=alpha)
        for pid, v in zip(wd.alive_ids, v_hat):
            records.append({
                "season": wd.season,
                "week": wd.week,
                "celebrity_name": pid,
                "v_hat": float(v),
                "is_eliminated_that_week": int(pid == wd.alive_ids[wd.eliminated_idx]),
            })

    out = pd.DataFrame(records).sort_values(["season", "week", "v_hat"], ascending=[True, True, False])
    return out


if __name__ == "__main__":
    # TODO: 改成你的实际路径
    DATA_PATH = "2026_MCM_Problem_C_Data.csv"

    # 超参数：先给默认值，后面你可以调 w/beta/alpha 做敏感性分析
    out_df = run_season_11_map(DATA_PATH, w=0.5, beta=10.0, alpha=1.0)

    print(out_df.head(20))
    out_df.to_csv("season11_map_vhat.csv", index=False)
    print("Saved: season11_map_vhat.csv")
