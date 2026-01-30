import re
import numpy as np
import pandas as pd

# ========= 路径（改成你自己的） =========
VHAT_PATH = "season11_map_vhat.csv"
RAW_PATH  = "2026_MCM_Problem_C_Data.csv"

# ========= 超参数 =========
W = 0.5  # 裁判权重在百分比合并中的占比

_WEEK_JUDGE_RE = re.compile(r"^week(\d+)_judge(\d+)_score$")


def find_week_judge_columns(df: pd.DataFrame) -> dict[int, list[str]]:
    week_cols: dict[int, list[str]] = {}
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


def compute_judge_totals(season_df: pd.DataFrame, week_cols: dict[int, list[str]]) -> pd.DataFrame:
    out = season_df[["celebrity_name", "season"]].copy()

    for w, cols in week_cols.items():
        numeric = season_df[cols].apply(pd.to_numeric, errors="coerce")
        out[f"J_week{w}"] = numeric.sum(axis=1, skipna=True)

        all_nan_mask = numeric.isna().all(axis=1)
        out.loc[all_nan_mask, f"J_week{w}"] = np.nan

    return out


def build_weekly_detail_table(
    vhat_df: pd.DataFrame,
    judge_totals_df: pd.DataFrame,
    w: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = sorted(vhat_df["week"].unique().tolist())
    jt = judge_totals_df.set_index("celebrity_name")

    detail_rows = []
    weekly_rows = []

    for wk in weeks:
        sub = vhat_df[vhat_df["week"] == wk].copy()

        alive = sub["celebrity_name"].tolist()
        Jcol = f"J_week{wk}"
        if Jcol not in jt.columns:
            raise ValueError(f"Raw judge totals missing column: {Jcol}. Check week range / column names.")

        sub["judge_total"] = [jt.loc[name, Jcol] if name in jt.index else np.nan for name in alive]
        sub = sub.dropna(subset=["judge_total"]).copy()
        if sub.empty:
            continue

        denom = sub["judge_total"].sum()
        if denom <= 0:
            continue

        sub["judge_share"] = sub["judge_total"] / denom

        sub["fan_share"] = pd.to_numeric(sub["v_hat"], errors="coerce")

        sub["judge_rank"] = sub["judge_share"].rank(method="average", ascending=False)
        sub["fan_rank"]   = sub["fan_share"].rank(method="average", ascending=False)

        sub["score_pct"] = float(w) * sub["judge_share"] + (1.0 - float(w)) * sub["fan_share"]
        sub["score_pct_rank"] = sub["score_pct"].rank(method="average", ascending=False)

        sub["rank_sum"] = sub["judge_rank"] + sub["fan_rank"]
        sub["rank_sum_rank"] = sub["rank_sum"].rank(method="average", ascending=True)

        elim_pct_name  = sub.loc[sub["score_pct"].idxmin(), "celebrity_name"]
        elim_rank_name = sub.loc[sub["rank_sum"].idxmax(), "celebrity_name"]

        sub["elim_by_percentage"] = (sub["celebrity_name"] == elim_pct_name).astype(int)
        sub["elim_by_rank"]       = (sub["celebrity_name"] == elim_rank_name).astype(int)

        if "is_eliminated_that_week" in sub.columns:
            sub["true_eliminated"] = pd.to_numeric(sub["is_eliminated_that_week"], errors="coerce").fillna(0).astype(int)
        else:
            sub["true_eliminated"] = 0

        sub_out = sub[[
            "season", "week", "celebrity_name",
            "judge_total", "judge_share", "judge_rank",
            "fan_share", "fan_rank",
            "score_pct", "score_pct_rank",
            "rank_sum", "rank_sum_rank",
            "true_eliminated",
            "elim_by_percentage",
            "elim_by_rank",
        ]].copy()

        detail_rows.append(sub_out)

        weekly_rows.append({
            "season": int(sub_out["season"].iloc[0]),
            "week": int(wk),
            "eliminated_by_percentage": elim_pct_name,
            "eliminated_by_rank": elim_rank_name,
            "same_or_not": int(elim_pct_name == elim_rank_name),
        })

    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    weekly_compare_df = pd.DataFrame(weekly_rows)

    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            ["season", "week", "fan_share"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

    return detail_df, weekly_compare_df


def main():
    # -------- Load v_hat (task1 output) --------
    vhat = pd.read_csv(VHAT_PATH)
    vhat["season"] = pd.to_numeric(vhat["season"], errors="coerce").astype(int)
    vhat["week"]   = pd.to_numeric(vhat["week"], errors="coerce").astype(int)
    vhat["v_hat"]  = pd.to_numeric(vhat["v_hat"], errors="coerce")

    # Season 11 only
    vhat11 = vhat[vhat["season"] == 11].copy()
    if vhat11.empty:
        raise ValueError("No season=11 rows found in season11_map_vhat.csv. Check file contents.")

    # -------- Load raw data to compute judge totals --------
    raw = pd.read_csv(RAW_PATH)
    if "season" not in raw.columns or "celebrity_name" not in raw.columns:
        raise ValueError("Raw CSV must contain columns: season, celebrity_name")

    raw["season"] = pd.to_numeric(raw["season"], errors="coerce")
    raw = raw[raw["season"].notna()].copy()
    raw["season"] = raw["season"].astype(int)

    season11_raw = raw[raw["season"] == 11].copy()
    if season11_raw.empty:
        raise ValueError("No season=11 rows found in raw data CSV. Check RAW_PATH and season column.")

    week_cols = find_week_judge_columns(season11_raw)
    if not week_cols:
        raise ValueError("No columns matching weekX_judgeY_score found in raw CSV.")

    jt = compute_judge_totals(season11_raw, week_cols)

    # -------- Build tables --------
    detail_df, weekly_compare_df = build_weekly_detail_table(vhat11, jt, w=W)

    # Output 1: full detail table
    detail_df.to_csv("season11_weekly_detail_table.csv", index=False)
    print("Saved: season11_weekly_detail_table.csv")

    # Output 2: weekly elimination comparison
    weekly_compare_df.to_csv("season11_rule_compare_weekly.csv", index=False)
    print("Saved: season11_rule_compare_weekly.csv")


if __name__ == "__main__":
    main()
