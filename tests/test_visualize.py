import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from matplotlib.axes import Axes

import src.visualize as visualize


def test_visualize_multiple_projects_with_envelopes(monkeypatch, tmp_path, capsys):
    class FakeRun:
        def __init__(
            self, entity, project, run_id, name, metric_name, values, runtime=600
        ):
            self.entity = entity
            self.project = project
            self.id = run_id
            self.name = name
            self.summary = {"_runtime": runtime}
            self.updated_at = datetime.utcnow()
            self._history_df = pd.DataFrame(
                {"_step": list(range(len(values))), metric_name: values}
            )

        def history(self, keys=None, pandas=False, samples=None):
            df = self._history_df.copy()
            if keys:
                cols = ["_step"] + [key for key in keys if key in df.columns]
                df = df[cols]
            return df if pandas else df.to_dict(orient="records")

    # Provide multiple runs per project/env to produce min/max envelopes, and two environments.
    project_runs = {
        "entityA/projectA": [
            FakeRun(
                "entityA",
                "projectA",
                "runA1",
                "CartPole_alpha1",
                "metric_A",
                [0.1, 0.2, 0.3],
            ),
            FakeRun(
                "entityA",
                "projectA",
                "runA2",
                "CartPole_alpha2",
                "metric_A",
                [0.05, 0.25, 0.35],
            ),
            FakeRun(
                "entityA",
                "projectA",
                "runA5",
                "CartPole_alpha3",
                "metric_A",
                [10, 20, 30],
            ),
            FakeRun(
                "entityA",
                "projectA",
                "runA3",
                "MountainCar_x",
                "metric_A",
                [-120, -100, -80],
            ),
            FakeRun(
                "entityA",
                "projectA",
                "runA6",
                "HalfCheetah_x",
                "metric_A",
                [-120, -100, -80],
            ),
        ],
        "entityB/projectB": [
            FakeRun(
                "entityB",
                "projectB",
                "runB1",
                "CartPole_beta1",
                "metric_B",
                [0.15, 0.18, 0.28],
            ),
            FakeRun(
                "entityB",
                "projectB",
                "runB2",
                "CartPole_beta2",
                "metric_B",
                [1.05, 2.22, 3.4],
            ),
            FakeRun(
                "entityB",
                "projectB",
                "runB3",
                "MountainCar_y",
                "metric_B",
                [-110, -90, -70],
            ),
        ],
    }

    class FakeApi:
        def __init__(self, timeout):
            self.timeout = timeout

        def runs(self, project_path):
            return project_runs.get(project_path, [])

    monkeypatch.setattr(visualize.wandb, "Api", FakeApi)

    # Capture fill_between calls on Axes to verify envelope painting.
    calls = []

    def fake_fill_between(self, x, y1, y2, *args, **kwargs):
        calls.append({"ax": self, "x": x, "y1": y1, "y2": y2, "kwargs": kwargs})
        return None

    monkeypatch.setattr(Axes, "fill_between", fake_fill_between)

    output_path = tmp_path / "grid.png"
    cache_dir = tmp_path / "cache"
    argv = [
        "visualize",
        "--project-metric",
        "entityA/projectA::metric_A",
        "--project-metric",
        "entityB/projectB::metric_B",
        "--cache-dir",
        str(cache_dir),
        "--output",
        str(output_path),
        "--smoothing-window",
        "1",
        "--show-individual",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    visualize.main()
    captured = capsys.readouterr()

    # basic file output checks
    assert output_path.exists() and output_path.stat().st_size > 0
    assert "from 2 project(s)" in captured.out

    # ensure per-project run counts were reported (pluralized by the script)
    assert "entityA/projectA [metric_A] -> 5 runs" in captured.out
    assert "entityB/projectB [metric_B] -> 3 runs" in captured.out

    # we should have captured fill_between calls; at least one should be the min/max envelope (alpha ~ 0.08)
    assert len(calls) > 0

    # collect alpha values seen (if any) and accept any small-alpha fill as the envelope
    alphas = [c["kwargs"].get("alpha") for c in calls if "alpha" in c["kwargs"]]
    has_minmax = any(a is not None and float(a) <= 0.2 for a in alphas)
    assert (
        has_minmax
    ), "Did not find any min/max envelope fill_between calls (alpha<=0.1)"

    # Expect fills to have been drawn on multiple axes (multiple environments)
    unique_axes = {id(c["ax"]) for c in calls}
    assert len(unique_axes) >= 1
