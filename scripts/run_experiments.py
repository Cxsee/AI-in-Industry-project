"""Run the plots notebook headlessly, resuming its unchanged checkpoint protocol."""

import argparse
import json
import os
from pathlib import Path


def execute_cells(path: Path, scope: dict, *, budget: int | None = None) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        # Colab mounting/install/checkout is unnecessary and unsafe in a local runner.
        if cell["cell_type"] != "code" or index == 1:
            continue
        exec(compile("".join(cell["source"]), f"{path.name} cell {index}", "exec"), scope)
        if budget is not None and index == 2:
            scope["MAX_MOVIES_PER_RUN"] = budget


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--max-movies", type=int, default=100, help="Attempt budget, including retries"
    )
    parser.add_argument("--run-content-after", action="store_true")
    args = parser.parse_args()
    if args.max_movies < 1:
        parser.error("--max-movies must be positive")
    for name, path in [
        ("MOVIEREC_DATA_DIR", args.data_dir),
        ("MOVIEREC_RESULTS_DIR", args.results_dir),
    ]:
        if path is not None:
            os.environ[name] = str(path.resolve())
    notebooks = Path(__file__).resolve().parents[1] / "notebooks"
    scope = {"__name__": "__plots_run__"}
    remaining = args.max_movies
    budget = min(40, remaining)
    execute_cells(notebooks / "01b_plots.ipynb", scope, budget=budget)
    while True:
        remaining -= budget
        plots = scope["checkpoint_results"](scope["selected"], scope["checkpoint_root"])
        unfinished = (
            scope["selected"].height
            - plots.filter(scope["pl"].col("status") != scope["RETRYABLE_STATUS"]).height
        )
        if not unfinished:
            print("Plot extraction complete.", flush=True)
            if args.run_content_after:
                execute_cells(notebooks / "04_content_based.ipynb", {"__name__": "__content_run__"})
                execute_cells(notebooks / "07_comparison.ipynb", {"__name__": "__comparison_run__"})
            return 0
        if plots.filter(scope["pl"].col("status") == scope["RETRYABLE_STATUS"]).height:
            print(
                "Stopped with retryable errors; inspect plots_progress.json, then rerun.",
                flush=True,
            )
            return 2
        if remaining <= 0:
            print("Attempt budget reached. Rerun the same command to resume.", flush=True)
            return 0
        budget = min(40, remaining)
        scope["MAX_MOVIES_PER_RUN"] = budget
        scope["process_pending"](scope["selected"], scope["checkpoint_root"])
        scope["export_plots"](
            scope["checkpoint_results"](scope["selected"], scope["checkpoint_root"]),
            scope["selected"],
            scope["checkpoint_root"],
        )


if __name__ == "__main__":
    raise SystemExit(main())
