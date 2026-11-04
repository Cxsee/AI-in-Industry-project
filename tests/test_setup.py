"""Checks that the environment works end to end, before any real code exists."""

import polars as pl
from ortools.sat.python import cp_model

from movierec import paths


def test_paths_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("MOVIEREC_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == tmp_path


def test_polars_reads_back_parquet(tmp_path):
    frame = pl.DataFrame({"movie_id": [1, 2], "rating": [4.0, 3.5]})
    frame.write_parquet(tmp_path / "ratings.parquet")
    assert pl.read_parquet(tmp_path / "ratings.parquet").equals(frame)


def test_cp_sat_picks_best_movies_under_a_constraint():
    scores = {"a": 90, "b": 80, "c": 10}
    long_tail = {"c"}

    model = cp_model.CpModel()
    picked = {movie: model.new_bool_var(movie) for movie in scores}
    model.add(sum(picked.values()) == 2)
    model.add(sum(picked[movie] for movie in long_tail) >= 1)
    model.maximize(sum(score * picked[movie] for movie, score in scores.items()))

    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert {movie for movie, var in picked.items() if solver.value(var)} == {"a", "c"}
