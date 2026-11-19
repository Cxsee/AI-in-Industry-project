"""Exercise the notebook's real checkpoint/export workflow without remote services."""

import ast
import json
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def workflow(tmp_path):
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "01b_plots.ipynb").read_text(encoding="utf-8")
    )
    scope = {}
    exec("".join(notebook["cells"][2]["source"]), scope)
    scope["DATA_DIR"] = tmp_path
    exec("".join(notebook["cells"][4]["source"]), scope)
    for index in [5, 6, 7, 8, 9]:
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)]
        exec(compile(tree, f"plots cell {index}", "exec"), scope)
    scope["SCOPE"] = "experiments"
    scope["RATINGS_PATH"] = None
    scope["USE_ARCHIVE_RATINGS"] = False
    scope["imdb_matches"] = lambda _: {}
    scope["wikidata_entities"] = lambda _: {}
    return scope


def test_processed_catalog_checkpoint_resume_and_export(workflow, tmp_path):
    w = workflow
    processed = pl.DataFrame(
        {"movieId": [1, 2, 3], "title": ["A", "B", "C"], "imdbId": [114709, 113497, 113228]}
    )
    processed.write_parquet(tmp_path / "movies.parquet")
    population = w["load_catalog"]()
    assert population["imdbId"].to_list() == ["tt0114709", "tt0113497", "tt0113228"]
    attempted = []

    def extract(movie, matches, entities):
        attempted.append(movie["movieId"])
        return w["result"](movie, "no_plot_section")

    w["extract_movie"] = extract
    root = tmp_path / "checkpoints"
    w["MAX_MOVIES_PER_RUN"] = 1
    w["process_pending"](population, root)
    partial = w["checkpoint_results"](population, root)
    w["export_plots"](partial, population, root)
    assert not (tmp_path / "plots.parquet").exists()
    progress = json.loads((tmp_path / "plots_progress.json").read_text())
    assert progress["pending_movies"] == 2
    assert progress["terminal_fallback_movies"] == 1
    w["MAX_MOVIES_PER_RUN"] = 10
    w["process_pending"](population, root)
    complete = w["checkpoint_results"](population, root)
    w["export_plots"](complete, population, root)
    assert attempted == [1, 2, 3]
    assert pl.read_parquet(tmp_path / "plots.parquet").height == 3
    assert pl.read_parquet(tmp_path / "plots_fallback.parquet").height == 3
    report = json.loads((tmp_path / "plots_coverage.json").read_text())
    assert report["movies_needing_fallback"] == 3
    assert report["coverage"] == 0


def test_retryable_records_are_retried_without_publishing(workflow, tmp_path):
    w = workflow
    population = pl.DataFrame({"movieId": [1], "title": ["A"], "imdbId": ["tt0114709"]})
    root = tmp_path / "checkpoints"
    w["extract_movie"] = lambda movie, *_: w["result"](movie, "retryable_error", detail="HTTP 503")
    w["process_pending"](population, root)
    w["export_plots"](w["checkpoint_results"](population, root), population, root)
    progress = json.loads((tmp_path / "plots_progress.json").read_text())
    assert progress["retryable_movies"] == 1
    assert progress["terminal_fallback_movies"] == 0
    assert not (tmp_path / "plots.parquet").exists()
    w["extract_movie"] = lambda movie, *_: w["result"](movie, "no_plot_section")
    w["process_pending"](population, root)
    complete = w["checkpoint_results"](population, root)
    assert complete["status"].to_list() == ["no_plot_section"]


def test_multilingual_provenance_verification(workflow):
    w = workflow
    population = pl.DataFrame({"movieId": [1], "title": ["A"], "imdbId": ["tt0114709"]})
    row = w["result"](
        population.row(0, named=True),
        "ok",
        plot="Una trama " * 20,
        quality="automatic_candidate",
        wikidata_id="Q1",
        plot_language="it",
        plot_wikipedia_url="https://it.wikipedia.org/wiki/A",
        plot_revision_id=123,
        plot_revision_url="https://it.wikipedia.org/w/index.php?oldid=123",
        plot_section="Trama",
        plot_chars=200,
    )
    frame = pl.DataFrame([row], schema=w["RESULT_SCHEMA"])
    w["verify_dataset"](frame, population)
    with pytest.raises(ValueError, match="provenance"):
        w["verify_dataset"](frame.with_columns(pl.lit(199).alias("plot_chars")), population)
    assert w["normalize_heading"]("  TRAMA :  ") == "trama"
    assert "あらすじ" in w["LOCAL_HEADINGS"]["ja"]


def test_full_article_plot_section_excludes_lead_tables_and_subsections(workflow):
    html = """<p>Lead must be excluded.</p>
    <div class="mw-heading"><h2 id="Plot">Plot</h2></div>
    <p>Narrative one.<sup>citation</sup></p>
    <table><tr><td><p>Cast must be excluded.</p></td></tr></table>
    <p>Narrative two.</p><h3>Analysis</h3><p>Not narrative.</p>
    <h2>Cast</h2><p>Another section.</p>"""
    assert workflow["section_prose"](html, "Plot") == "Narrative one. Narrative two."
    old_heading = '<h2><span id="Trama">Trama</span></h2><p>Una storia.</p>'
    assert workflow["section_prose"](old_heading, "Trama") == "Una storia."
    with pytest.raises(workflow["RemoteError"], match="anchor"):
        workflow["section_prose"](html, "missing")
