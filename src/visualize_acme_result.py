import argparse
import math
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument(
        "--x-column",
        type=str,
        default="evaluator_step",
        help=(
            "Column to use for the x-axis. If set to 'evaluator_step' and that column "
            "is missing, the script will fall back to 'evaluator_steps'. "
            "If neither exists, the row index is used."
        ),
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
        "--graphs-per-row",
        dest="graphs_per_row",
        type=int,
        default=4,
        help="Number of subplots (graphs) per row (columns). Must be >= 1.",
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

    file_path = args.file
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Determine x-axis values
    x_col = args.x_column
    if (
        x_col == "evaluator_step"
        and "evaluator_step" not in df.columns
        and "evaluator_steps" in df.columns
    ):
        x_col = "evaluator_steps"

    if x_col in df.columns:
        x_vals = pd.to_numeric(df[x_col], errors="coerce")
        if x_vals.isna().all():
            x_vals = pd.Series(range(len(df)))
    else:
        x_vals = pd.Series(range(len(df)))

    x_vals = x_vals.reset_index(drop=True)

    # Identify numeric columns to plot (exclude the x column)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != x_col]

    if not numeric_cols:
        raise ValueError(
            "No numeric columns found to plot (after excluding the x-column)."
        )

    series_list = []  # list of (col_name, x_vals, s_raw, s_smoothed)

    for col in numeric_cols:
        s_raw = pd.to_numeric(df[col], errors="coerce").reset_index(drop=True)
        if s_raw.isna().all():
            continue

        if args.smooth_window and args.smooth_window > 1:
            s_smoothed = s_raw.rolling(
                window=args.smooth_window, min_periods=1, center=True
            ).mean()
        else:
            s_smoothed = s_raw

        series_list.append((col, x_vals, s_raw, s_smoothed))

    if not series_list:
        raise ValueError("No valid numeric columns to plot.")

    # Grid layout
    n_plots = len(series_list)
    n_cols = max(1, int(args.graphs_per_row))
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

    # Global x-limits (shared across plots)
    all_x_min = x_vals.min()
    all_x_max = x_vals.max()

    # Plot each numeric column
    for i, (col_name, x_vals, s_raw, s_sm) in enumerate(series_list):
        ax = axes[i]
        x = x_vals.values

        if args.show_raw:
            ax.plot(x, s_raw.values, marker="o", linestyle=":", alpha=0.4, label="raw")

        ax.plot(
            x,
            s_sm.values,
            marker="o" if args.smooth_window <= 1 else None,
            label="smoothed",
        )

        ax.set_title(col_name)
        ax.set_xlabel(x_col if x_col in df.columns else "Row index")
        ax.set_ylabel(col_name)
        if args.show_raw or args.smooth_window:
            ax.legend(fontsize="small")

        ax.set_xlim(all_x_min, all_x_max)

    # Hide unused axes (if any)
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
