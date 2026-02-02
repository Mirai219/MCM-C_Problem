import re
from dataclasses import dataclass
from typing import Dict, List

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
    alive_ids: List[str]
    judge_share: np.ndarray
    eliminated_idx: int


# =========================
# 2) 解析列名：weekX_judgeY_score
# =========================
_WEEK_JUDGE_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")


def find_week_judge_columns(df: pd.DataFrame) -> Dict[int, List[str]]:
    week_cols: Dict[int, List[str]] = {}
    for c in df.columns:
        m = _WEEK_JUDGE_RE.match(c)
        if m:
            w = int(m.group(1))
            week_cols.setdefault(w, []).append(c)

    for w in week_cols:
        week_cols[w] = sorted(
            week_cols[w],
            key=lambda x: int(_WEEK_JUDGE_RE.match(x).group(2))
        )

    return dict(sorted(week_cols.items(), key=lambda kv: kv[0]))


def compute_judge_totals_by_week(
    season_df: pd.DataFrame,
    week_cols: Dict[int, List[str]]
) -> pd.DataFrame:

    out = season_df[['celebrity_name', 'season']].copy()

    for w, cols in week_cols.items():
        numeric = season_df[cols].apply(pd.to_numeric, errors='coerce')
        out[f"J_week{w}"] = numeric.sum(axis=1, skipna=True)

        all_nan_mask = numeric.isna().all(axis=1)
        out.loc[all_nan_mask, f"J_week{w}"] = np.nan

    return out


# =========================
# 3) 构造 WeekData（season=11）
# =========================
def build_week_data_for_season_11(df: pd.DataFrame) -> List[WeekData]:
    season = 11
    season_df = df[df['season'] == season].copy()
    if season_df.empty:
        raise ValueError("season=11 数据为空")

    week_cols = find_week_judge_columns(season_df)
    jt = compute_judge_totals_by_week(season_df, week_cols)

    weeks = sorted(
        int(re.search(r"J_week(\d+)", c).group(1))
        for c in jt.columns if c.startswith("J_week")
    )
    max_week = max(weeks)

    week_data_list: List[WeekData] = []

    for w in range(1, max_week):
        cur_col = f"J_week{w}"
        nxt_col = f"J_week{w+1}"

        cur = jt[cur_col]
        nxt = jt[nxt_col]

        alive_mask = (cur.notna()) & (cur > 0)
        alive_df = jt.loc[alive_mask, ['celebrity_name', cur_col, nxt_col]].copy()
        if alive_df.empty:
            continue

        eliminated_mask = (alive_df[nxt_col].notna()) & (alive_df[nxt_col] == 0)
        eliminated = alive_df.loc[eliminated_mask, 'celebrity_name'].tolist()

        if len(eliminated) != 1:
            continue

        alive_ids = alive_df['celebrity_name'].tolist()
        eliminated_idx = alive_ids.index(eliminated[0])

        judge_totals = alive_df[cur_col].to_numpy(dtype=float)
        denom = judge_totals.sum()
        if denom <= 0:
            continue

        judge_share = judge_totals / denom

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
        raise ValueError("未构造出有效 WeekData")

    return week_data_list


# =========================
# 4) 单周 MAP（PyTorch）
# =========================
def solve_week_map(
    week_data: WeekData,
    w: float = 0.5,
    beta: float = 10.0,
    alpha: float = 1.0,
    max_iter: int = 200,
    seed: int = 0,
) -> np.ndarray:

    torch.manual_seed(seed)

    judge_share = torch.tensor(week_data.judge_share, dtype=torch.double)
    n = judge_share.numel()
    elim = int(week_data.eliminated_idx)

    z = torch.zeros(n, dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([z], lr=1.0, max_iter=max_iter)

    alpha_tensor = torch.full((n,), float(alpha), dtype=torch.double)
    eps = 1e-12

    def closure():
        optimizer.zero_grad()

        v = torch.softmax(z, dim=0)
        S = w * judge_share + (1 - w) * v

        logits = -beta * S
        loglik = torch.log_softmax(logits, dim=0)[elim]
        logprior = torch.sum((alpha_tensor - 1) * torch.log(v + eps))

        loss = -(loglik + logprior)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        return torch.softmax(z, dim=0).cpu().numpy()


# =========================
# 5) 全流程（含裁判分数扰动）
# =========================
def run_season_11_map(
    csv_path: str,
    w: float = 0.5,
    beta: float = 10.0,
    alpha: float = 1.0,
    judge_scale: float = 1.0,   # ⭐ 敏感度参数
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    # 裁判分数统一缩放
    judge_cols = [c for c in df.columns if _WEEK_JUDGE_RE.match(c)]
    if judge_cols:
        df[judge_cols] = df[judge_cols] * judge_scale

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"].notna()].copy()
    df["season"] = df["season"].astype(int)

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

    return (
        pd.DataFrame(records)
        .sort_values(["season", "week", "v_hat"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


# =========================
# 6) 主程序：±5% 敏感度分析
# =========================
if __name__ == "__main__":
    DATA_PATH = "2026_MCM_Problem_C_Data.csv"

    for scale, tag in [
        (0.95, "minus5"),
        (1.00, "baseline"),
        (1.05, "plus5"),
    ]:
        out_df = run_season_11_map(
            DATA_PATH,
            w=0.5,
            beta=10.0,
            alpha=1.0,
            judge_scale=scale
        )

        fname = f"season11_map_vhat_{tag}.csv"
        out_df.to_csv(fname, index=False)
        print(f"Saved: {fname}")
