"""
python plot_metric_grid.py \
  --metric episode_return \
  --file "Enb 0::runs/env0.csv" \
  --file "Enb 1::runs/env1.csv" \
  --file "Enb 2::runs/env2.csv"
"""

import argparse
import math
import os

import pandas as pd
import matplotlib.pyplot as plt


def parse_file_arg(file_arg: str):
    """
    Parse 'name::path' into (name, path).
    """
    if "::" not in file_arg:
        raise ValueError(
            f'Invalid --file argument "{file_arg}". '
            'Expected format: "name::path/to/file.csv"'
        )
    name, path = file_arg.split("::", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f'Empty name in --file argument: "{file_arg}"')
    if not path:
        raise ValueError(f'Empty path in --file argument: "{file_arg}"')
    return name, path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a single metric from multiple CSV files, "
            "one subplot per file (2 plots per row), with shared axis scales."
        )
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        required=True,
        help='CSV file spec in the form "name::path/to/file.csv". '
        "Can be given multiple times.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Name of the metric (column) to plot from each CSV.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=14.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height_per_row",
        type=float,
        default=3.0,
        help="Figure height per row in inches.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help=(
            "Apply centered rolling mean smoothing with this window size (integer > 1). "
            "0 means no smoothing. Odd window is recommended."
        ),
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Overlay raw data (faint) on top of the smoothed line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/acme/metric_plot.png",
        help="If set, save the figure to this path instead of (or in addition to) showing it.",
    )

    args = parser.parse_args()

    metric = args.metric
    series_list = []  # list of (name, x_series, raw_series, smoothed_series)
    has_evaluator_steps = False

    # ---------- load data ----------
    for file_arg in args.files:
        name, path = parse_file_arg(file_arg)

        if not os.path.isfile(path):
            raise FileNotFoundError(f'File for "{name}" not found: {path}')

        df = pd.read_csv(path)

        if metric not in df.columns:
            raise ValueError(
                f'Metric "{metric}" not found in file "{path}". '
                f"Available columns: {list(df.columns)}"
            )

        s_raw = pd.to_numeric(df[metric], errors="coerce")
        if s_raw.isna().all():
            raise ValueError(
                f'Metric "{metric}" in file "{path}" is not numeric or all NaN.'
            )

        # determine x axis values: prefer `evaluator_steps` column if present and numeric
        if "evaluator_steps" in df.columns:
            x_vals = pd.to_numeric(df["evaluator_steps"], errors="coerce")
            if not x_vals.isna().all():
                has_evaluator_steps = True
            else:
                # fallback to integer index
                x_vals = pd.Series(range(len(df)))
        else:
            x_vals = pd.Series(range(len(df)))

        # reset indices so x, raw and smoothed series align by position
        x_vals = x_vals.reset_index(drop=True)
        s_raw = s_raw.reset_index(drop=True)

        # apply centered rolling mean if requested
        if args.smooth_window and args.smooth_window > 1:
            s_smoothed = s_raw.rolling(
                window=args.smooth_window, min_periods=1, center=True
            ).mean()
        else:
            s_smoothed = s_raw

        series_list.append((name, x_vals, s_raw, s_smoothed))

    if not series_list:
        raise ValueError("No valid files/metrics to plot.")

    # ---------- compute global scales ----------
    # x-axis: use evaluator_steps (if present) or integer index. Compute global x limits.
    all_x_min = min(x_vals.min() for _, x_vals, _, _ in series_list)
    all_x_max = max(x_vals.max() for _, x_vals, _, _ in series_list)

    # y-axis: based on metric values across all files
    # compute y-limits from the smoothed series (matches plotted lines)
    y_min_global = min(s_sm.min() for _, _, _, s_sm in series_list)
    y_max_global = max(s_sm.max() for _, _, _, s_sm in series_list)
    y_range = y_max_global - y_min_global

    if y_range == 0:
        # all values same across all series; create a symmetric window around that value
        base = max(abs(y_max_global), 1.0)
        y_pad = 0.25 * base
        y_min_global = y_min_global - y_pad
        y_max_global = y_max_global + y_pad
    else:
        y_pad = 0.25 * y_range
        y_min_global = y_min_global - y_pad
        y_max_global = y_max_global + y_pad

    # ---------- prepare grid ----------
    n_plots = len(series_list)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(args.width, args.height_per_row * n_rows),
    )

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # ---------- plot ----------
    for i, (name, x_vals, s_raw, s_sm) in enumerate(series_list):
        ax = axes[i]
        x = x_vals.values
        if args.show_raw:
            ax.plot(x, s_raw.values, marker="o", linestyle=":", alpha=0.4, label="raw")
        # plot smoothed series
        ax.plot(
            x,
            s_sm.values,
            marker="o" if args.smooth_window <= 1 else None,
            label="smoothed",
        )
        ax.set_title(name)
        ax.set_xlabel("evaluator_steps" if has_evaluator_steps else "Row index")
        ax.set_ylabel(metric)
        if args.show_raw or args.smooth_window:
            ax.legend(fontsize="small")

        # apply global limits
        ax.set_xlim(all_x_min, all_x_max)
        ax.set_ylim(y_min_global, y_max_global)

    # hide unused axes
    for j in range(len(series_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if args.output:
        out_path = args.output
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
