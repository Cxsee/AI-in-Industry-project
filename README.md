# AI-in-Industry-project

Movie recommendations beyond accuracy: we compare classic recommenders, an LLM used as a ranker and
a hybrid with a constrained selection step (OR-Tools CP-SAT), on accuracy, novelty, diversity,
exposure and cost.

Andrea Caselli and Francesco Forlani · AI in Industry exam project + Project Work

## How we work

We write and run the code on Google Colab and version it on GitHub. Datasets and results stay on
Google Drive.

```text
src/movierec/   code shared between notebooks
notebooks/      one notebook per activity, each with one owner
tests/          tests
data/           datasets (local runs only, not committed)
results/        outputs (local runs only, not committed)
```

## First setup on Colab

1. Create a GitHub token that can read this repo.
2. In Colab, add it under **Secrets** (key icon) with the name `GITHUB_TOKEN`.
3. Open [`notebooks/00_colab_setup.ipynb`](https://colab.research.google.com/github/Cxsee/AI-in-Industry-project/blob/main/notebooks/00_colab_setup.ipynb)
   and run it. All the tests should pass.
4. Copy its setup cell at the top of every new notebook.

Everything that isn't on Drive is lost when the Colab session ends, so save model outputs and LLM
answers to `paths.results_dir()` as you go.

## Saving your work

- **Notebooks:** use **File → Save a copy in GitHub** and pick your branch. Create the branch on
  GitHub first.
- **Clear the outputs before saving:** **Edit → Clear all outputs**. Save the plots you need to
  `paths.results_dir()`.
- **Code in `src/movierec/`:** edit it on GitHub or locally, then run `git pull` in the notebook.
- **Changes to `main`:** go through a pull request, reviewed by the other person.

Two people editing the same notebook produce conflicts that are hard to merge, so each notebook has
one owner. Notebooks exchange data through Parquet files on Drive.

## Optional extras

| Extra | Used for |
| --- | --- |
| `mf` | Matrix factorization (`implicit`) |
| `content` | Content-based embeddings (`sentence-transformers`) |
| `llm` | LLM ranker through Ollama |

On Colab, set them in `EXTRAS` in the setup cell.

## Running locally

You need [uv](https://docs.astral.sh/uv/); it installs Python by itself.

```bash
uv sync --extra mf
uv run pytest
uv run ruff check .
uv run ruff format .
```

`uv sync` removes whatever you don't ask for, so pass every extra you need each time.

## Validation comparison and Content-Based

`07_comparison` evaluates the required `popularity_val.parquet` and `mf_val.parquet`
files on the same saved users with `evaluate_validation`. It adds `content_val.parquet`
when available. Outputs include `comparison_val.csv`, a wide comparison Parquet,
a combined summary, and per-model/per-user metrics under `paths.results_dir()`.

Wikipedia through Wikidata is the agreed plot source (confirmed 18 November 2026).
`01b_plots` now defaults to `experiments` and reads `experiment_movies.parquet`.
It can use the processed `movies.parquet`, so the MovieLens archive is optional.
Start with a small attempt budget, inspect `plots_progress.json`, then resume the
same checkpoint directory. Terminal missing plots and transient errors are distinct.
Only a complete extraction writes `plots.parquet`, `plots_coverage.json` and
`plots_fallback.parquet`; the coverage report includes languages and fallback counts.

A local runner uses the same notebook code and checkpoints. Set `MOVIEREC_DATA_DIR`
and `MOVIEREC_RESULTS_DIR` if datasets are outside the repository:

```powershell
uv run --extra content python scripts/run_experiments.py --max-movies 100
# After checking the small run, complete extraction and then run Content-Based/comparison:
uv run --extra content python scripts/run_experiments.py --max-movies 30000 --run-content-after
```

The extractor retrieves text, Wikidata identity and revision in one parse response,
then selects only the explicit plot section. Requests stay serial and respect API
backoff. The runner stops on retryable service failures. Inspect the progress/error message and rerun the same command to resume.
Do not run two extraction processes against the same checkpoint directory.

`04_content_based` uses pinned multilingual sentence embeddings, chunk pooling for
long plots, mean liked-film profiles and cosine ranking. Tags are restricted to
strictly before their author's last fitting rating. Validation fits on train; test
uses train + validation and requires the configuration saved in validation. Missing
plots use genres/tags; users without a usable liked-film profile receive a recorded
Popularity fallback. Read the notebook and `decisions.md` for the protocol and limits.
