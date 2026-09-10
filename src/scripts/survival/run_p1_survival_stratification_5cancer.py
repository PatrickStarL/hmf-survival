#!/usr/bin/env python3
"""
Formal P1 survival stratification analysis from reused mainline results.

This script does not retrain anything. It reuses the completed 5-cancer
mainline run, loads each fold's out-of-fold test predictions, and builds:

- seed-level 5-fold OOF risk tables and KM plots
- 5-seed mean-risk ensemble OOF tables and KM plots
- log-rank statistics for each cancer
- a combined 5-cancer panel figure for result presentation
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["STAD", "CRC", "KIRC", "LUAD", "HNSC"]
DEFAULT_FOLDS = [0, 1, 2, 3, 4]


@dataclass
class SeedStat:
    dataset: str
    seed: int
    n_cases: int
    n_low_risk: int
    n_high_risk: int
    seed_count_used: int
    fold_count_used: int
    mean_fold_c_index_test: float
    std_fold_c_index_test: float
    oof_c_index: float
    risk_cutoff: float
    risk_split_note: str
    merge_note: str
    logrank_chi2: float
    logrank_p: float
    median_survival_low: float | None
    median_survival_high: float | None
    oof_csv: str
    figure_png: str
    figure_pdf: str


@dataclass
class EnsembleStat:
    dataset: str
    n_cases: int
    n_low_risk: int
    n_high_risk: int
    seed_count_expected: int
    seed_count_min_per_case: int
    seed_count_max_per_case: int
    mean_seed_oof_c_index: float
    std_seed_oof_c_index: float
    mean_seed_fold_c_index_test: float
    std_seed_fold_c_index_test: float
    ensemble_oof_c_index: float
    risk_cutoff: float
    risk_split_note: str
    merge_note: str
    logrank_chi2: float
    logrank_p: float
    median_survival_low: float | None
    median_survival_high: float | None
    oof_csv: str
    figure_png: str
    figure_pdf: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal P1 survival stratification from reused mainline results.")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--mainline-run-root", type=Path, default=None)
    parser.add_argument(
        "--mainline-pattern",
        type=str,
        default="/path/to/your/data_root/mainline_twostage_panther_mstar_conch_5cancer_*",
        help="Glob used when --mainline-run-root is omitted.",
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--folds", nargs="*", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--target-col", type=str, default="dss_survival_days")
    parser.add_argument("--time-unit", choices=["days", "years"], default="years")
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--generate-seed-plots", type=int, default=1)
    parser.add_argument("--generate-ensemble-plots", type=int, default=1)
    parser.add_argument("--generate-panel", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def latest_matching_path(pattern: str) -> Path | None:
    matches = sorted(Path("/").glob(pattern.lstrip("/")), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def resolve_mainline_run_root(args: argparse.Namespace) -> Path:
    if args.mainline_run_root is not None:
        return args.mainline_run_root
    found = latest_matching_path(args.mainline_pattern)
    if found is None:
        raise FileNotFoundError(
            f"Could not resolve mainline run root from pattern: {args.mainline_pattern}"
        )
    return found


def resolve_run_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        return args.run_root
    ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    return Path(f"/path/to/your/data_root/formal_p1_survival_stratification_5cancer_{ts}")


def safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def fmt_metric(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_pvalue(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 1e-3:
        return f"{value:.3g}"
    return f"{value:.4f}"


def fmt_surv(value: float | None, unit: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NR"
    digits = 2 if unit == "years" else 1
    return f"{value:.{digits}f}"


def concordance_index_censored(event_observed, event_time, estimate, tied_tol: float = 1e-8) -> float:
    event_observed = np.asarray(event_observed).astype(bool)
    event_time = np.asarray(event_time).astype(float)
    estimate = np.asarray(estimate).astype(float)

    concordant = 0.0
    permissible = 0.0
    ties = 0.0

    n = len(event_time)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = event_time[i], event_time[j]
            ei, ej = event_observed[i], event_observed[j]
            ri, rj = estimate[i], estimate[j]

            if ti == tj:
                continue
            if ti < tj and ei:
                permissible += 1.0
                diff = ri - rj
                if abs(diff) <= tied_tol:
                    ties += 1.0
                elif diff > 0:
                    concordant += 1.0
            elif tj < ti and ej:
                permissible += 1.0
                diff = rj - ri
                if abs(diff) <= tied_tol:
                    ties += 1.0
                elif diff > 0:
                    concordant += 1.0

    return (concordant + 0.5 * ties) / permissible if permissible > 0 else 0.5


def logrank_test(time, event, high_risk_group) -> Tuple[float, float]:
    time = np.asarray(time).astype(float)
    event = np.asarray(event).astype(int)
    high_risk_group = np.asarray(high_risk_group).astype(bool)

    event_times = np.sort(np.unique(time[event == 1]))
    if len(event_times) == 0:
        return 0.0, 1.0

    score = 0.0
    var = 0.0
    for t in event_times:
        at_risk = time >= t
        events_now = (time == t) & (event == 1)

        n_high = int(np.sum(at_risk & high_risk_group))
        n_low = int(np.sum(at_risk & (~high_risk_group)))
        d_high = int(np.sum(events_now & high_risk_group))
        d_low = int(np.sum(events_now & (~high_risk_group)))

        n_all = n_high + n_low
        d_all = d_high + d_low
        if n_all <= 1 or d_all <= 0:
            continue

        exp_high = d_all * (n_high / n_all)
        if n_all == 1:
            v_high = 0.0
        else:
            v_high = (n_high * n_low * d_all * (n_all - d_all)) / (
                (n_all ** 2) * (n_all - 1)
            )
        score += d_high - exp_high
        var += v_high

    if var <= 0:
        return 0.0, 1.0

    chi2 = float((score ** 2) / var)
    pvalue = float(math.erfc(math.sqrt(chi2 / 2.0)))
    return chi2, pvalue


def build_km_table(time, event) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": np.asarray(time).astype(float),
            "event": np.asarray(event).astype(int),
        }
    ).sort_values("time", kind="mergesort")

    if df.empty:
        return pd.DataFrame(columns=["time", "n_at_risk", "n_event", "n_censor", "survival_after"])

    rows = []
    surv = 1.0
    n_at_risk = int(len(df))
    for current_time in np.sort(df["time"].unique()):
        mask = df["time"] == current_time
        n_event = int(df.loc[mask, "event"].sum())
        n_total = int(mask.sum())
        n_censor = int(n_total - n_event)
        if n_event > 0 and n_at_risk > 0:
            surv = surv * (1.0 - n_event / n_at_risk)
        rows.append(
            {
                "time": float(current_time),
                "n_at_risk": int(n_at_risk),
                "n_event": int(n_event),
                "n_censor": int(n_censor),
                "survival_after": float(surv),
            }
        )
        n_at_risk -= n_total
    return pd.DataFrame(rows)


def km_step_arrays(km_table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = [0.0]
    y = [1.0]
    prev_surv = 1.0
    for row in km_table.itertuples(index=False):
        if int(row.n_event) <= 0:
            continue
        current_time = float(row.time)
        current_surv = float(row.survival_after)
        x.extend([current_time, current_time])
        y.extend([prev_surv, current_surv])
        prev_surv = current_surv
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def km_censor_points(km_table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for row in km_table.itertuples(index=False):
        if int(row.n_censor) <= 0:
            continue
        xs.extend([float(row.time)] * int(row.n_censor))
        ys.extend([float(row.survival_after)] * int(row.n_censor))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def km_median_survival(km_table: pd.DataFrame) -> float | None:
    for row in km_table.itertuples(index=False):
        if float(row.survival_after) <= 0.5:
            return float(row.time)
    return None


def decode_sample_id(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_prediction_dump(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pickle.load(f)

    required_keys = {"all_risk_scores", "all_censorships", "all_event_times", "sample_ids"}
    missing = required_keys.difference(obj.keys())
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")

    risk = np.asarray(obj["all_risk_scores"]).reshape(-1)
    censorship = np.asarray(obj["all_censorships"]).reshape(-1)
    event_time = np.asarray(obj["all_event_times"]).reshape(-1)
    sample_ids = np.asarray([decode_sample_id(x) for x in np.asarray(obj["sample_ids"]).reshape(-1)])

    n = len(sample_ids)
    if not (len(risk) == len(censorship) == len(event_time) == n):
        raise ValueError(
            f"Inconsistent dump lengths in {path}: "
            f"risk={len(risk)} censorship={len(censorship)} time={len(event_time)} ids={n}"
        )

    df = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "risk_score": risk.astype(float),
            "censorship": censorship.astype(float),
            "event_time_days": event_time.astype(float),
        }
    )
    df["event_observed"] = 1.0 - df["censorship"]
    return df


def load_done_status_rows(
    mainline_run_root: Path,
    datasets: Iterable[str],
    folds: Iterable[int],
) -> pd.DataFrame:
    status_tsv = mainline_run_root / "reports" / "run_status.tsv"
    if not status_tsv.is_file():
        raise FileNotFoundError(f"Missing mainline status table: {status_tsv}")

    df = pd.read_csv(status_tsv, sep="\t")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")
    df["c_index_test"] = pd.to_numeric(df["c_index_test"], errors="coerce")
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.sort_values("timestamp_utc")

    latest = df.groupby(["dataset", "seed", "fold"], dropna=False, as_index=False).tail(1).copy()
    done = latest[latest["status"] == "DONE"].copy()
    done = done[done["dataset"].isin(list(datasets))].copy()
    done = done[done["fold"].isin(list(folds))].copy()
    if done.empty:
        raise RuntimeError(f"No DONE rows found under {status_tsv}")

    done["seed"] = done["seed"].astype(int)
    done["fold"] = done["fold"].astype(int)
    done["result_dir"] = done["summary_csv"].map(lambda x: str(Path(str(x)).parent))
    done["test_results_pkl"] = done["summary_csv"].map(lambda x: str(Path(str(x)).parent / "test_results.pkl"))
    missing_dump = [p for p in done["test_results_pkl"].tolist() if not Path(p).is_file()]
    if missing_dump:
        raise FileNotFoundError(f"Missing test_results.pkl files, first few: {missing_dump[:5]}")
    return done.sort_values(["dataset", "seed", "fold"]).reset_index(drop=True)


def combine_seed_oof_rows(run_rows: pd.DataFrame, expected_folds: Iterable[int]) -> Tuple[pd.DataFrame, str]:
    frames = []
    for row in run_rows.itertuples(index=False):
        dump_df = load_prediction_dump(Path(row.test_results_pkl))
        dump_df["dataset"] = str(row.dataset)
        dump_df["seed"] = int(row.seed)
        dump_df["fold"] = int(row.fold)
        dump_df["task"] = str(row.task)
        frames.append(dump_df)

    merged = pd.concat(frames, ignore_index=True)
    present_folds = sorted(int(x) for x in merged["fold"].unique().tolist())
    expected_folds = sorted(int(x) for x in expected_folds)
    if present_folds != expected_folds:
        raise RuntimeError(f"Seed OOF fold mismatch: expected={expected_folds} got={present_folds}")

    dup_counts = merged["sample_id"].value_counts()
    duplicate_ids = dup_counts[dup_counts > 1].index.tolist()
    merge_note = "unique_sample_ids"
    if duplicate_ids:
        subset = merged[merged["sample_id"].isin(duplicate_ids)].copy()
        label_check = subset.groupby("sample_id")[["event_time_days", "censorship", "event_observed"]].nunique(dropna=False)
        bad = label_check[(label_check > 1).any(axis=1)]
        if not bad.empty:
            raise RuntimeError(
                "Duplicate sample_id with inconsistent survival labels detected: "
                f"{bad.index.tolist()[:10]}"
            )
        merged = (
            merged.groupby(["dataset", "seed", "sample_id"], as_index=False)
            .agg(
                risk_score=("risk_score", "mean"),
                censorship=("censorship", "first"),
                event_observed=("event_observed", "first"),
                event_time_days=("event_time_days", "first"),
            )
        )
        merge_note = f"dedup_mean_risk_for_{len(duplicate_ids)}_sample_ids"
    else:
        merged = merged.sort_values("sample_id").reset_index(drop=True)

    return merged, merge_note


def combine_ensemble_oof_rows(seed_tables: List[pd.DataFrame], expected_seed_count: int) -> Tuple[pd.DataFrame, str, int, int]:
    if not seed_tables:
        raise RuntimeError("No seed tables provided for ensemble.")

    combined = pd.concat(seed_tables, ignore_index=True)
    label_check = combined.groupby("sample_id")[["event_time_days", "censorship", "event_observed"]].nunique(dropna=False)
    bad = label_check[(label_check > 1).any(axis=1)]
    if not bad.empty:
        raise RuntimeError(
            "Sample label mismatch across seeds detected: "
            f"{bad.index.tolist()[:10]}"
        )

    grouped = (
        combined.groupby("sample_id", as_index=False)
        .agg(
            dataset=("dataset", "first"),
            risk_score=("risk_score", "mean"),
            censorship=("censorship", "first"),
            event_observed=("event_observed", "first"),
            event_time_days=("event_time_days", "first"),
            seed_count=("seed", "nunique"),
        )
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    seed_count_min = int(grouped["seed_count"].min())
    seed_count_max = int(grouped["seed_count"].max())
    merge_note = "all_cases_have_all_seeds"
    if seed_count_min != expected_seed_count or seed_count_max != expected_seed_count:
        merge_note = (
            f"seed_count_varied_min_{seed_count_min}_max_{seed_count_max}_expected_{expected_seed_count}"
        )
    return grouped, merge_note, seed_count_min, seed_count_max


def assign_risk_groups(df: pd.DataFrame, min_group_size: int) -> Tuple[pd.DataFrame, float, str]:
    out = df.copy()
    cutoff = float(np.median(out["risk_score"].to_numpy(dtype=float)))
    high_mask = out["risk_score"].to_numpy(dtype=float) >= cutoff
    note = "median_ge_cutoff"

    n_high = int(np.sum(high_mask))
    n_low = int(len(high_mask) - n_high)
    if min(n_high, n_low) < min_group_size:
        order = np.argsort(out["risk_score"].to_numpy(dtype=float), kind="mergesort")
        high_mask = np.zeros(len(out), dtype=bool)
        high_mask[order[len(out) // 2 :]] = True
        note = "balanced_rank_median_fallback"

    out["risk_group"] = np.where(high_mask, "High risk", "Low risk")
    out = out.sort_values(["risk_group", "risk_score", "sample_id"], ascending=[True, False, True]).reset_index(drop=True)
    return out, cutoff, note


def plot_payload_from_grouped_df(df: pd.DataFrame, time_unit: str) -> dict:
    scale = 365.25 if time_unit == "years" else 1.0
    xlabel = "Disease-specific survival time (years)" if time_unit == "years" else "Disease-specific survival time (days)"

    payload = {
        "time_unit": time_unit,
        "xlabel": xlabel,
        "curves": {},
        "xmax": 0.0,
    }
    for group_name in ["Low risk", "High risk"]:
        group_df = df[df["risk_group"] == group_name].copy()
        time = group_df["event_time_days"].to_numpy(dtype=float) / scale
        event = group_df["event_observed"].to_numpy(dtype=float).astype(int)
        km_table = build_km_table(time, event)
        step_x, step_y = km_step_arrays(km_table)
        censor_x, censor_y = km_censor_points(km_table)
        median_surv = km_median_survival(km_table)
        payload["curves"][group_name] = {
            "step_x": step_x,
            "step_y": step_y,
            "censor_x": censor_x,
            "censor_y": censor_y,
            "median_survival": median_surv,
            "n_cases": int(len(group_df)),
        }
        if len(time) > 0:
            payload["xmax"] = max(payload["xmax"], float(np.max(time)))
    return payload


def render_km_plot(
    ax,
    plot_payload: dict,
    title: str,
    subtitle_lines: List[str],
    show_legend: bool = True,
    show_censors: bool = True,
) -> None:
    colors = {
        "Low risk": "#2D7F84",
        "High risk": "#C84C61",
    }
    for group_name in ["Low risk", "High risk"]:
        curve = plot_payload["curves"][group_name]
        label = f"{group_name} (n={curve['n_cases']})"
        ax.step(
            curve["step_x"],
            curve["step_y"],
            where="post",
            color=colors[group_name],
            linewidth=2.2,
            label=label,
        )
        if show_censors and len(curve["censor_x"]) > 0:
            ax.scatter(
                curve["censor_x"],
                curve["censor_y"],
                marker="|",
                s=28,
                linewidths=1.0,
                color=colors[group_name],
                alpha=0.65,
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(plot_payload["xlabel"], fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, max(plot_payload["xmax"] * 1.03, 0.1))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.2)
    if show_legend:
        ax.legend(loc="lower left", frameon=False, fontsize=9)

    info_text = "\n".join(subtitle_lines)
    if info_text:
        ax.text(
            0.98,
            0.98,
            info_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D0D0D0", alpha=0.92),
        )


def save_single_km_figure(
    plot_payload: dict,
    dataset: str,
    label: str,
    subtitle_lines: List[str],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    render_km_plot(
        ax=ax,
        plot_payload=plot_payload,
        title=f"{dataset} | {label}",
        subtitle_lines=subtitle_lines,
        show_legend=True,
        show_censors=True,
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_ensemble_panel(
    panel_payloads: Dict[str, dict],
    datasets: List[str],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.2))
    axes = axes.reshape(-1)
    for ax in axes:
        ax.axis("off")

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ax.axis("on")
        payload = panel_payloads[dataset]
        render_km_plot(
            ax=ax,
            plot_payload=payload["plot_payload"],
            title=dataset,
            subtitle_lines=payload["subtitle_lines"],
            show_legend=True,
            show_censors=False,
        )

    fig.suptitle(
        "P1 Survival Stratification from Reused Mainline OOF Risk\n5-seed mean-risk ensemble over 5-fold mainline predictions",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def summarize_seed_rows(seed_rows: List[SeedStat]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in seed_rows]).sort_values(["dataset", "seed"]).reset_index(drop=True)


def summarize_ensemble_rows(ensemble_rows: List[EnsembleStat]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in ensemble_rows]).sort_values(["dataset"]).reset_index(drop=True)


def to_markdown(
    datasets: List[str],
    mainline_run_root: Path,
    seed_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    panel_png: Path | None,
    time_unit: str,
) -> str:
    lines = [
        "# P1 Survival Stratification Summary",
        "",
        f"- mainline_run_root: {mainline_run_root}",
        "- reuse_mode: direct reuse of completed mainline 5-seed x 5-fold test predictions",
        "- official_display: 5-seed mean-risk ensemble OOF KM plot for each cancer",
        "- split_rule: median risk split with balanced fallback when needed",
        f"- time_unit_for_plots: {time_unit}",
        "",
        "## Ensemble Figures",
        "",
        "| Dataset | Cases | Low risk | High risk | Ensemble OOF c-index | Log-rank p | Median survival low | Median survival high | Figure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for dataset in datasets:
        row_df = ensemble_df[ensemble_df["dataset"] == dataset]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        lines.append(
            f"| {dataset} | {int(row['n_cases'])} | {int(row['n_low_risk'])} | {int(row['n_high_risk'])} | "
            f"{fmt_metric(safe_float(row['ensemble_oof_c_index']))} | {fmt_pvalue(safe_float(row['logrank_p']))} | "
            f"{fmt_surv(safe_float(row['median_survival_low']), time_unit)} | {fmt_surv(safe_float(row['median_survival_high']), time_unit)} | "
            f"{row['figure_png']} |"
        )

    if panel_png is not None:
        lines.extend(
            [
                "",
                "## Combined Panel",
                "",
                f"- panel_png: {panel_png}",
                "",
            ]
        )

    lines.extend(
        [
            "## Seed-Level Reference",
            "",
            "| Dataset | Seed | Mean fold c-index | OOF c-index | Log-rank p | Merge note |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for dataset in datasets:
        subset = seed_df[seed_df["dataset"] == dataset].copy()
        if subset.empty:
            continue
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {int(row.seed)} | {fmt_metric(safe_float(row.mean_fold_c_index_test))} | "
                f"{fmt_metric(safe_float(row.oof_c_index))} | {fmt_pvalue(safe_float(row.logrank_p))} | {row.merge_note} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    datasets = list(args.datasets)
    folds = [int(x) for x in args.folds]
    mainline_run_root = resolve_mainline_run_root(args)
    run_root = resolve_run_root(args)

    report_root = run_root / "reports"
    tables_root = run_root / "tables"
    figures_root = run_root / "figures"
    report_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    done_df = load_done_status_rows(mainline_run_root, datasets=datasets, folds=folds)
    used_runs_csv = report_root / "mainline_done_runs_used.csv"
    done_df.to_csv(used_runs_csv, index=False)

    seed_group_df = (
        done_df.groupby(["dataset", "seed"], dropna=False)["c_index_test"]
        .agg(mean_fold_c_index_test="mean", std_fold_c_index_test=lambda s: s.std(ddof=0), fold_count_used="count")
        .reset_index()
        .sort_values(["dataset", "seed"])
        .reset_index(drop=True)
    )

    seed_rows: List[SeedStat] = []
    ensemble_rows: List[EnsembleStat] = []
    panel_payloads: Dict[str, dict] = {}

    for dataset in datasets:
        dataset_done = done_df[done_df["dataset"] == dataset].copy()
        if dataset_done.empty:
            raise RuntimeError(f"No DONE rows found for dataset={dataset}")

        dataset_seed_group = seed_group_df[seed_group_df["dataset"] == dataset].copy()
        seeds = [int(x) for x in dataset_seed_group["seed"].tolist()]
        if not seeds:
            raise RuntimeError(f"No completed seeds found for dataset={dataset}")

        print(
            f"[dataset] start dataset={dataset} seeds={','.join(str(s) for s in seeds)} folds={','.join(str(f) for f in folds)}",
            flush=True,
        )

        dataset_seed_tables: List[pd.DataFrame] = []
        dataset_seed_oof_cindices: List[float] = []
        dataset_seed_foldmean_cindices: List[float] = []

        for seed in seeds:
            seed_runs = dataset_done[dataset_done["seed"] == seed].copy()
            seed_oof_df, merge_note = combine_seed_oof_rows(seed_runs, expected_folds=folds)
            seed_oof_df, cutoff, split_note = assign_risk_groups(seed_oof_df, min_group_size=args.min_group_size)

            chi2, pvalue = logrank_test(
                time=seed_oof_df["event_time_days"].to_numpy(dtype=float),
                event=seed_oof_df["event_observed"].to_numpy(dtype=float),
                high_risk_group=(seed_oof_df["risk_group"] == "High risk").to_numpy(),
            )
            oof_c_index = concordance_index_censored(
                event_observed=seed_oof_df["event_observed"].to_numpy(dtype=float).astype(bool),
                event_time=seed_oof_df["event_time_days"].to_numpy(dtype=float),
                estimate=seed_oof_df["risk_score"].to_numpy(dtype=float),
            )

            plot_payload = plot_payload_from_grouped_df(seed_oof_df, time_unit=args.time_unit)
            low_med = plot_payload["curves"]["Low risk"]["median_survival"]
            high_med = plot_payload["curves"]["High risk"]["median_survival"]

            seed_oof_dir = tables_root / "seed_oof" / dataset
            seed_oof_dir.mkdir(parents=True, exist_ok=True)
            seed_oof_csv = seed_oof_dir / f"{dataset}_seed_{seed}_oof.csv"
            seed_oof_df.to_csv(seed_oof_csv, index=False)

            seed_png = figures_root / "seed_oof" / dataset / f"km_{dataset}_seed_{seed}_oof.png"
            seed_pdf = figures_root / "seed_oof" / dataset / f"km_{dataset}_seed_{seed}_oof.pdf"

            mean_fold_cidx = safe_float(
                dataset_seed_group.loc[dataset_seed_group["seed"] == seed, "mean_fold_c_index_test"].iloc[0]
            )
            std_fold_cidx = safe_float(
                dataset_seed_group.loc[dataset_seed_group["seed"] == seed, "std_fold_c_index_test"].iloc[0]
            )
            fold_count_used = int(
                dataset_seed_group.loc[dataset_seed_group["seed"] == seed, "fold_count_used"].iloc[0]
            )

            subtitle_lines: List[str] = []
            if args.generate_seed_plots:
                save_single_km_figure(
                    plot_payload=plot_payload,
                    dataset=dataset,
                    label=f"seed {seed} OOF",
                    subtitle_lines=subtitle_lines,
                    png_path=seed_png,
                    pdf_path=seed_pdf,
                    dpi=args.dpi,
                )

            seed_rows.append(
                SeedStat(
                    dataset=dataset,
                    seed=seed,
                    n_cases=int(len(seed_oof_df)),
                    n_low_risk=int((seed_oof_df["risk_group"] == "Low risk").sum()),
                    n_high_risk=int((seed_oof_df["risk_group"] == "High risk").sum()),
                    seed_count_used=1,
                    fold_count_used=fold_count_used,
                    mean_fold_c_index_test=mean_fold_cidx,
                    std_fold_c_index_test=std_fold_cidx,
                    oof_c_index=float(oof_c_index),
                    risk_cutoff=float(cutoff),
                    risk_split_note=split_note,
                    merge_note=merge_note,
                    logrank_chi2=float(chi2),
                    logrank_p=float(pvalue),
                    median_survival_low=None if low_med is None else float(low_med),
                    median_survival_high=None if high_med is None else float(high_med),
                    oof_csv=str(seed_oof_csv),
                    figure_png=str(seed_png),
                    figure_pdf=str(seed_pdf),
                )
            )

            dataset_seed_tables.append(seed_oof_df[["dataset", "seed", "sample_id", "risk_score", "censorship", "event_observed", "event_time_days"]].copy())
            dataset_seed_oof_cindices.append(float(oof_c_index))
            dataset_seed_foldmean_cindices.append(float(mean_fold_cidx))

            print(
                "[seed] "
                f"dataset={dataset} seed={seed} "
                f"oof_c_index={oof_c_index:.6f} "
                f"logrank_p={pvalue:.6g}",
                flush=True,
            )

        ensemble_oof_df, ensemble_merge_note, seed_count_min, seed_count_max = combine_ensemble_oof_rows(
            dataset_seed_tables,
            expected_seed_count=len(seeds),
        )
        ensemble_oof_df, ensemble_cutoff, ensemble_split_note = assign_risk_groups(
            ensemble_oof_df,
            min_group_size=args.min_group_size,
        )

        ensemble_chi2, ensemble_pvalue = logrank_test(
            time=ensemble_oof_df["event_time_days"].to_numpy(dtype=float),
            event=ensemble_oof_df["event_observed"].to_numpy(dtype=float),
            high_risk_group=(ensemble_oof_df["risk_group"] == "High risk").to_numpy(),
        )
        ensemble_oof_c_index = concordance_index_censored(
            event_observed=ensemble_oof_df["event_observed"].to_numpy(dtype=float).astype(bool),
            event_time=ensemble_oof_df["event_time_days"].to_numpy(dtype=float),
            estimate=ensemble_oof_df["risk_score"].to_numpy(dtype=float),
        )

        ensemble_plot_payload = plot_payload_from_grouped_df(ensemble_oof_df, time_unit=args.time_unit)
        ensemble_low_med = ensemble_plot_payload["curves"]["Low risk"]["median_survival"]
        ensemble_high_med = ensemble_plot_payload["curves"]["High risk"]["median_survival"]

        ensemble_oof_dir = tables_root / "ensemble_oof"
        ensemble_oof_dir.mkdir(parents=True, exist_ok=True)
        ensemble_oof_csv = ensemble_oof_dir / f"{dataset}_ensemble_oof.csv"
        ensemble_oof_df.to_csv(ensemble_oof_csv, index=False)

        ensemble_png = figures_root / "ensemble" / f"km_{dataset}_ensemble_oof.png"
        ensemble_pdf = figures_root / "ensemble" / f"km_{dataset}_ensemble_oof.pdf"
        ensemble_subtitle_lines: List[str] = []
        if args.generate_ensemble_plots:
            save_single_km_figure(
                plot_payload=ensemble_plot_payload,
                dataset=dataset,
                label="5-seed mean-risk ensemble OOF",
                subtitle_lines=ensemble_subtitle_lines,
                png_path=ensemble_png,
                pdf_path=ensemble_pdf,
                dpi=args.dpi,
            )

        ensemble_rows.append(
            EnsembleStat(
                dataset=dataset,
                n_cases=int(len(ensemble_oof_df)),
                n_low_risk=int((ensemble_oof_df["risk_group"] == "Low risk").sum()),
                n_high_risk=int((ensemble_oof_df["risk_group"] == "High risk").sum()),
                seed_count_expected=int(len(seeds)),
                seed_count_min_per_case=int(seed_count_min),
                seed_count_max_per_case=int(seed_count_max),
                mean_seed_oof_c_index=float(np.mean(dataset_seed_oof_cindices)),
                std_seed_oof_c_index=float(np.std(dataset_seed_oof_cindices)) if len(dataset_seed_oof_cindices) > 1 else 0.0,
                mean_seed_fold_c_index_test=float(np.mean(dataset_seed_foldmean_cindices)),
                std_seed_fold_c_index_test=float(np.std(dataset_seed_foldmean_cindices)) if len(dataset_seed_foldmean_cindices) > 1 else 0.0,
                ensemble_oof_c_index=float(ensemble_oof_c_index),
                risk_cutoff=float(ensemble_cutoff),
                risk_split_note=ensemble_split_note,
                merge_note=ensemble_merge_note,
                logrank_chi2=float(ensemble_chi2),
                logrank_p=float(ensemble_pvalue),
                median_survival_low=None if ensemble_low_med is None else float(ensemble_low_med),
                median_survival_high=None if ensemble_high_med is None else float(ensemble_high_med),
                oof_csv=str(ensemble_oof_csv),
                figure_png=str(ensemble_png),
                figure_pdf=str(ensemble_pdf),
            )
        )

        panel_payloads[dataset] = {
            "plot_payload": ensemble_plot_payload,
            "subtitle_lines": [],
        }

        print(
            "[dataset] "
            f"dataset={dataset} "
            f"ensemble_oof_c_index={ensemble_oof_c_index:.6f} "
            f"ensemble_logrank_p={ensemble_pvalue:.6g}",
            flush=True,
        )

    seed_stat_df = summarize_seed_rows(seed_rows)
    ensemble_stat_df = summarize_ensemble_rows(ensemble_rows)

    seed_stats_csv = report_root / "p1_survival_stratification_seed_oof_stats.csv"
    ensemble_stats_csv = report_root / "p1_survival_stratification_ensemble_oof_stats.csv"
    seed_stat_df.to_csv(seed_stats_csv, index=False)
    ensemble_stat_df.to_csv(ensemble_stats_csv, index=False)

    panel_png = None
    panel_pdf = None
    if args.generate_panel:
        panel_png = figures_root / "ensemble" / "km_panel_5cancer_ensemble.png"
        panel_pdf = figures_root / "ensemble" / "km_panel_5cancer_ensemble.pdf"
        save_ensemble_panel(
            panel_payloads=panel_payloads,
            datasets=datasets,
            png_path=panel_png,
            pdf_path=panel_pdf,
            dpi=args.dpi,
        )

    summary_payload = {
        "generated_at_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_root": str(run_root),
        "mainline_run_root": str(mainline_run_root),
        "mainline_status_tsv": str(mainline_run_root / "reports" / "run_status.tsv"),
        "datasets": datasets,
        "folds": folds,
        "target_col": args.target_col,
        "time_unit": args.time_unit,
        "min_group_size": int(args.min_group_size),
        "used_runs_csv": str(used_runs_csv),
        "seed_stats_csv": str(seed_stats_csv),
        "ensemble_stats_csv": str(ensemble_stats_csv),
        "panel_png": None if panel_png is None else str(panel_png),
        "panel_pdf": None if panel_pdf is None else str(panel_pdf),
        "seed_oof_stats": seed_stat_df.to_dict(orient="records"),
        "ensemble_oof_stats": ensemble_stat_df.to_dict(orient="records"),
    }

    summary_json = report_root / "p1_survival_stratification_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = report_root / "p1_survival_stratification_summary.md"
    summary_md.write_text(
        to_markdown(
            datasets=datasets,
            mainline_run_root=mainline_run_root,
            seed_df=seed_stat_df,
            ensemble_df=ensemble_stat_df,
            panel_png=panel_png,
            time_unit=args.time_unit,
        ),
        encoding="utf-8",
    )

    run_config = report_root / "run_config.txt"
    run_config.write_text(
        "\n".join(
            [
                f"generated_at_utc={pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"run_root={run_root}",
                f"mainline_run_root={mainline_run_root}",
                f"mainline_status_tsv={mainline_run_root / 'reports' / 'run_status.tsv'}",
                f"datasets={' '.join(datasets)}",
                f"folds={' '.join(str(f) for f in folds)}",
                f"target_col={args.target_col}",
                f"time_unit={args.time_unit}",
                f"min_group_size={args.min_group_size}",
                f"generate_seed_plots={args.generate_seed_plots}",
                f"generate_ensemble_plots={args.generate_ensemble_plots}",
                f"generate_panel={args.generate_panel}",
                f"dpi={args.dpi}",
                "method=Reuse mainline 5-seed x 5-fold test predictions; assemble seed-level OOF risk then 5-seed mean-risk ensemble OOF; perform KM plus log-rank on each cancer.",
                "official_display=5-seed mean-risk ensemble OOF KM plots for STAD CRC KIRC LUAD HNSC",
                "risk_split=median_ge_cutoff with balanced rank-median fallback when a group is too small",
                f"used_runs_csv={used_runs_csv}",
                f"seed_stats_csv={seed_stats_csv}",
                f"ensemble_stats_csv={ensemble_stats_csv}",
                f"summary_json={summary_json}",
                f"summary_md={summary_md}",
                f"panel_png={'' if panel_png is None else panel_png}",
                f"panel_pdf={'' if panel_pdf is None else panel_pdf}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overall_mean = float(ensemble_stat_df["ensemble_oof_c_index"].mean()) if not ensemble_stat_df.empty else float("nan")
    overall_best_p = float(ensemble_stat_df["logrank_p"].min()) if not ensemble_stat_df.empty else float("nan")
    print(
        "[finished] "
        f"ensemble_mean_oof_c_index={fmt_metric(overall_mean)} "
        f"best_logrank_p={fmt_pvalue(overall_best_p)}",
        flush=True,
    )
    print(run_root)
    print(seed_stats_csv)
    print(ensemble_stats_csv)
    print(summary_json)
    print(summary_md)
    if panel_png is not None:
        print(panel_png)


if __name__ == "__main__":
    main()
