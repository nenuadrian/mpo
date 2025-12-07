import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def _build_project_path(project: str, entity: str | None) -> str:
    if "/" in project:
        return project
    if not entity:
        raise ValueError(
            "Please provide --entity when the project name has no entity prefix."
        )
    return f"{entity}/{project}"


def _run_timestamp(run: wandb.apis.public.Run) -> str:
    candidate = getattr(run, "updated_at", None)
    if hasattr(candidate, "isoformat"):
        return candidate.isoformat()
    if candidate:
        return str(candidate)
    summary_ts = run.summary.get("_timestamp") if hasattr(run, "summary") else None
    if summary_ts is not None:
        return str(summary_ts)
    return run.id


def _prepare_history_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame(columns=["step", metric])

    df = df.copy()
    step_col = None
    for candidate in ["_step", "step", "global_step", "training_iteration", "epoch"]:
        if candidate in df.columns:
            step_col = candidate
            break
    if step_col is None:
        df["step"] = range(len(df))
    else:
        df.rename(columns={step_col: "step"}, inplace=True)

    df = df[["step", metric]]
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["step", metric])
    df = df.sort_values("step")
    df = df.drop_duplicates(subset="step", keep="last")
    df.reset_index(drop=True, inplace=True)
    return df


def _load_history_with_cache(
    run: wandb.apis.public.Run,
    metric: str,
    cache_dir: Path,
    max_samples: int,
) -> pd.DataFrame:
    run_dir = cache_dir / run.entity / run.project / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_safe = metric.replace("/", "_")
    data_path = run_dir / f"{metric_safe}.csv"
    meta_path = run_dir / f"{metric_safe}.json"

    run_stamp = _run_timestamp(run)

    if data_path.exists() and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("updated_at") == run_stamp:
                cached = pd.read_csv(data_path)
                if cached.empty or metric not in cached.columns:
                    return pd.DataFrame(columns=["step", metric])
                cached["step"] = pd.to_numeric(cached["step"], errors="coerce")
                cached[metric] = pd.to_numeric(cached[metric], errors="coerce")
                cached = cached.dropna(subset=["step", metric])
                cached = cached.sort_values("step").reset_index(drop=True)
                return cached
        except Exception:
            pass

    history = run.history(keys=[metric], pandas=True, samples=max_samples)
    prepared = _prepare_history_frame(history, metric)
    prepared.to_csv(data_path, index=False)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"updated_at": run_stamp, "rows": int(len(prepared)), "metric": metric}, fh
        )
    return prepared


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if series.empty or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _aggregate_runs(
    dfs: List[pd.DataFrame], metric: str, smoothing_window: int = 1
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Aggregate multiple run DataFrames into per-step mean/std/min/max/count.
    If smoothing_window > 1, apply rolling smoothing to each run individually
    before computing aggregated statistics so envelopes align with the plotted mean.
    """
    series_list: List[pd.Series] = []
    for df in dfs:
        if df.empty or metric not in df.columns:
            continue
        s = pd.Series(df[metric].values, index=df["step"].values)
        s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()
        if smoothing_window and smoothing_window > 1:
            s = s.rolling(window=smoothing_window, min_periods=1).mean()
        series_list.append(s)
    if not series_list:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=int),
        )

    combined = pd.concat(series_list, axis=1, sort=True)
    combined.columns = range(len(series_list))
    combined = combined.sort_index()
    mean_series = combined.mean(axis=1, skipna=True)
    std_series = combined.std(axis=1, skipna=True)
    min_series = combined.min(axis=1, skipna=True)
    max_series = combined.max(axis=1, skipna=True)
    count_series = combined.count(axis=1)
    return mean_series, std_series, min_series, max_series, count_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one or more W&B scalars across runs grouped by environment."
    )
    parser.add_argument(
        "--project-metric",
        dest="project_metrics",
        action="append",
        required=True,
        help="Project/metric pair in the form 'project::metric'. May be specified multiple times.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (team or username). Optional if project already includes it.",
    )
    parser.add_argument(
        "--min-runtime-minutes",
        type=float,
        default=5.0,
        help="Keep runs whose runtime is at least this many minutes (default: 5).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=20,
        help="Window size for moving average smoothing (default: 20).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("logs/wandb_cache"),
        help="Directory used to cache downloaded histories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wandb_metric_grid.png"),
        help="Path to save the resulting grid figure (PNG).",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of columns in the plot grid (default: 3).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000000,
        help="Maximum number of history rows to request per run (default: 2000000).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="W&B API timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Overlay individual run curves in addition to the smoothed mean.",
    )

    args = parser.parse_args()

    project_inputs = args.project_metrics

    project_specs: List[Dict[str, str]] = []
    for item in project_inputs:
        try:
            project_raw, metric = item.split("::", 1)
        except ValueError:
            parser.error(
                f"Invalid --project-metric value '{item}'. Use the format 'project::metric'."
            )
        project_path = _build_project_path(project_raw, args.entity)
        project_specs.append(
            {"path": project_path, "metric": metric, "label": project_path}
        )

    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # deterministic ordered list of project labels (used to pick consistent colors)
    all_project_labels = sorted(spec["label"] for spec in project_specs)
    base_color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_color_cycle:
        base_color_cycle = [f"C{i}" for i in range(10)]
    project_color_map = {
        label: base_color_cycle[i % len(base_color_cycle)]
        for i, label in enumerate(all_project_labels)
    }

    api = wandb.Api(timeout=args.timeout)
    env_to_project_dfs: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
    project_metrics = {spec["label"]: spec["metric"] for spec in project_specs}
    project_kept_runs = defaultdict(int)

    min_runtime_seconds = args.min_runtime_minutes * 60.0
    kept_runs = 0
    for spec in project_specs:
        runs = list(api.runs(spec["path"]))
        for run in runs:
            runtime = float(run.summary.get("_runtime", 0.0))
            if runtime < min_runtime_seconds:
                continue
            run_name = run.name or run.id
            env_name = run_name.split("_")[0]
            df = _load_history_with_cache(
                run, spec["metric"], cache_dir, args.max_samples
            )
            if df.empty:
                continue
            env_to_project_dfs.setdefault(env_name, {}).setdefault(
                spec["label"], []
            ).append(df)
            project_kept_runs[spec["label"]] += 1
            kept_runs += 1

    if not env_to_project_dfs:
        print(
            "No data found for the requested metric(s). Check the inputs and try again."
        )
        return

    env_names = sorted(env_to_project_dfs.keys())
    n_envs = len(env_names)
    ncols = max(1, args.ncols)
    nrows = math.ceil(n_envs / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 4.5, nrows * 3.5),
        squeeze=False,
    )
    axes_iter = axes.ravel()

    for idx, env_name in enumerate(env_names):
        ax = axes_iter[idx]
        project_map = env_to_project_dfs[env_name]
        plotted = False
        total_runs = sum(len(dfs) for dfs in project_map.values())
        # iterate only projects that have data for this env, but use global colors
        for project_idx, project_label in enumerate(sorted(project_map)):
            dfs = project_map[project_label]
            metric = project_metrics[project_label]
            # aggregate runs; smooth per-run inside aggregation so envelopes match the mean
            mean_series, std_series, min_series, max_series, _ = _aggregate_runs(
                dfs, metric, smoothing_window=args.smoothing_window
            )
            if mean_series.empty:
                continue

            color = project_color_map.get(project_label, f"C{project_idx % 10}")

            if args.show_individual:
                for df in dfs:
                    ax.plot(
                        df["step"],
                        df[metric],
                        color=color,
                        alpha=0.2,
                        linewidth=1,
                    )

            ax.plot(
                mean_series.index,
                mean_series.values,
                color=color,
                linewidth=2,
                label=f"{project_label} ({metric})",
            )
            if not std_series.empty:
                std_bounds = pd.concat(
                    [
                        (mean_series - std_series).rename("lower"),
                        (mean_series + std_series).rename("upper"),
                    ],
                    axis=1,
                ).dropna()
                if not std_bounds.empty:
                    x_s = np.asarray(std_bounds.index.astype(float))
                    y_lower = std_bounds["lower"].to_numpy(dtype=float)
                    y_upper = std_bounds["upper"].to_numpy(dtype=float)
                    ax.fill_between(x_s, y_lower, y_upper, color=color, alpha=0.18)
            plotted = True

        ax.set_title(f"{env_name} ({total_runs} run{'s' if total_runs != 1 else ''})")
        ax.set_xlabel("step")
        ax.set_ylabel("metric value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for ax in axes_iter[n_envs:]:
        ax.set_visible(False)

    fig.tight_layout()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Processed {kept_runs} runs across {n_envs} environment(s) from {len(project_specs)} project(s)."
    )
    for spec in project_specs:
        print(
            f"  {spec['label']} [{spec['metric']}] -> {project_kept_runs[spec['label']]} run{'s' if project_kept_runs[spec['label']] != 1 else ''}"
        )
    print(f"Saved figure to {output_path}.")


if __name__ == "__main__":
    main()
