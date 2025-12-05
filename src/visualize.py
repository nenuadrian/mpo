import argparse
import os
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
import matplotlib.pyplot as plt


def _is_event_file(filename: str) -> bool:
    """Return True if filename looks like a TensorBoard event file."""
    # Typical names are like: events.out.tfevents.<timestamp>.<hostname>.<pid>
    # We'll use a simple and robust check.
    return "events.out.tfevents." in filename


def _find_event_files(root: str, grep: Optional[str] = None) -> List[str]:
    """Recursively find event files under root, optionally filtering by a grep substring.

    - root: directory to search
    - grep: if provided, only keep file paths containing this substring (case-sensitive,
      similar to `grep` default)
    """
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if _is_event_file(fname):
                fpath = os.path.join(dirpath, fname)
                if grep and grep not in fpath:
                    continue
                matches.append(fpath)
    return sorted(matches)


def _collect_scalars_from_files(
    files: Iterable[str], tags: Optional[List[str]]
) -> Tuple[pd.DataFrame, Set[str]]:
    """Read scalars from multiple event files and return a DataFrame and the union of tags.

    DataFrame columns: [file, tag, step, value]
    """
    all_rows = []
    union_tags: Set[str] = set()

    for fpath in files:
        try:
            ea = EventAccumulator(fpath)
            ea.Reload()
        except Exception as e:
            print(f"Warning: failed to load {fpath}: {e}")
            continue

        file_tags = ea.Tags().get("scalars", [])
        union_tags.update(file_tags)

        # If specific tags are requested, only use those that exist in this file
        tags_for_file = (
            [t for t in (tags or []) if t in file_tags] if tags else file_tags
        )
        for t in tags_for_file:
            try:
                for e in ea.Scalars(t):
                    all_rows.append(
                        {
                            "file": fpath,
                            "tag": t,
                            "step": e.step,
                            "value": e.value,
                        }
                    )
            except KeyError:
                # Tag missing in this particular file; skip.
                continue

    return pd.DataFrame(all_rows), union_tags


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse TensorBoard .tfevents: pass an existing path (file or dir) OR a "
            "grep-like substring to search for across all event files starting at $PWD."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        help=(
            "Either an existing path (file/dir) OR a substring to grep for in event file paths."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.getcwd(),
        help=(
            "Search root when using grep-style query (default: current working directory)."
        ),
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more scalar tags to extract (e.g. --tags rollout/ep_rew_mean train/loss). "
            "If omitted, extracts all scalar tags."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of scalar entries to print for preview (default: 5).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help=(
            "If provided, save one PNG per tag into this directory; otherwise show plots interactively."
        ),
    )
    args = parser.parse_args()

    # Resolve files based on whether query is a path or a grep substring
    query = args.query
    files: List[str] = []

    if os.path.exists(query):
        if os.path.isdir(query):
            files = _find_event_files(query)
        else:
            # If it's a file, use it directly; if it's a directory-ish TB path, _find_event_files will handle.
            files = [query]
    else:
        # Treat as grep-like substring under the provided root
        files = _find_event_files(args.root, grep=query)

    if not files:
        print(
            "No matching event files found. Check your query or --root. "
            "Example: query='Hopper' or query='/path/to/dir'"
        )
        return

    print(f"Found {len(files)} event file(s).")
    for f in files[:10]:  # don't spam; show up to 10
        print(f"  - {f}")
    if len(files) > 10:
        print("  ...")

    # Resolve requested tags (prefer --tags, fallback to --tag)
    requested_tags: Optional[List[str]] = None
    requested_tags = args.tags

    df, union_tags = _collect_scalars_from_files(files, tags=requested_tags)

    if not df.shape[0]:
        print("No scalar data found. Check your tag names or files.")
        if union_tags:
            print(f"Available scalar tags across files: {sorted(union_tags)}")
        return

    print(
        f"\nAvailable scalar tags across files ({len(union_tags)}):\n{sorted(union_tags)}\n"
    )

    # Show preview
    print(df.head(args.limit))

    # Plot: one figure per tag, all files overlaid
    tags_to_plot = requested_tags if requested_tags else sorted(union_tags)
    if not tags_to_plot:
        return

    os.makedirs(args.save_dir, exist_ok=True)

    def _sanitize_filename(name: str) -> str:
        return (
            name.replace(os.sep, "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    for t in tags_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sub = df[df["tag"] == t]
        if sub.empty:
            plt.close(fig)
            continue
        for fpath, g in sub.groupby("file"):
            g = g.sort_values("step")
            label = os.path.basename(fpath).replace("events.out.tfevents.", "")
            ax.plot(g["step"], g["value"], label=label, linewidth=1.5)
        ax.set_title(t)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=1)

        out_path = os.path.join(args.save_dir, f"{_sanitize_filename(t)}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
