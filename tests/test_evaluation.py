from math import log2

import polars as pl
import pytest

from movierec.evaluation import evaluate_validation


def evaluate(rows, *, k=2):
    ratings = pl.LazyFrame(
        [
            (1, 11, 2.0, "train"),
            (2, 12, 5.0, "train"),
            (3, 12, 5.0, "train"),
            (4, 12, 5.0, "train"),
            (4, 13, 3.0, "train"),
            (4, 14, 5.0, "train"),
            (1, 13, 4.0, "val"),
            (1, 99, 5.0, "val"),
            (2, 13, 3.5, "val"),
            (3, 13, 4.5, "val"),
            (1, 14, 5.0, "test"),
            (2, 99, 5.0, "test"),
        ],
        schema=["userId", "movieId", "rating", "split"],
        orient="row",
    )
    users = pl.DataFrame({"userId": [3, 1, 2]})
    movies = pl.DataFrame(
        {"movieId": [11, 12, 13, 14, 99], "genres": [[], ["Drama"], ["Comedy"], [], []]}
    )
    recommendations = pl.DataFrame(
        rows, schema={"userId": pl.Int64, "movieId": pl.Int64, "rank": pl.Int64}, orient="row"
    )
    return evaluate_validation(ratings, users, movies, recommendations, k=k, min_ratings=1)


def test_validation_cohort_metrics_and_aggregation():
    per_user, summary = evaluate([(1, 13, 2), (1, 12, 1), (2, 14, 1)])
    assert per_user["userId"].to_list() == [3, 1, 2]
    assert per_user["n_positives"].to_list() == [1, 2, 0]
    assert per_user["recall"].to_list() == [0.0, 0.5, None]
    assert per_user["ndcg"][1] == pytest.approx(1 / (1 + log2(3)))
    result = summary.row(0, named=True)
    assert result["recall"] == 0.25
    assert result["ndcg"] == pytest.approx(1 / (2 * (1 + log2(3))))
    assert result["n_users"] == 3
    assert result["n_accuracy_users"] == 2
    assert result["n_excluded_no_positives"] == 1
    assert result["n_empty_rankings"] == 1
    assert result["novelty"] == pytest.approx(((log2(4 / 3) + 2) / 2 + 2) / 3)
    assert result["genre_diversity"] == pytest.approx(1 / 3)
    assert result["long_tail_share"] == pytest.approx(1 / 3)
    assert result["catalog_coverage"] == 0.75


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(1, 12, 1), (1, 12, 2)], "duplicate"),
        ([(1, 11, 1)], "already rated"),
        ([(1, 99, 1)], "outside the eligible catalog"),
        ([(4, 12, 1)], "outside the evaluation sample"),
        ([(1, 12, 2)], "consecutive ranks"),
        ([(1, 12, 1), (1, 13, 1)], "consecutive ranks"),
        ([(1, 12, 1), (1, 13, 3)], "consecutive ranks"),
        ([(1, 12, 1), (1, 13, 2), (1, 14, 3)], "at most 2"),
        ([(1, None, 1)], "non-null integers"),
    ],
)
def test_rejects_invalid_recommendations(rows, message):
    with pytest.raises(ValueError, match=message):
        evaluate(rows)


def test_accepts_an_empty_ranking_file_with_schema():
    per_user, summary = evaluate([])
    assert per_user["recall"].to_list() == [0.0, 0.0, None]
    assert summary["n_empty_rankings"].item() == 3
    assert summary["catalog_coverage"].item() == 0.0
