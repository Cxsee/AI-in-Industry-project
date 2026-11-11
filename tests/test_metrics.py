from math import log2

import polars as pl
import pytest

from movierec.metrics import (
    catalog_coverage,
    embedding_diversity_at_k,
    genre_diversity_at_k,
    long_tail_share_at_k,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)


def test_recall_uses_all_distinct_positives_and_only_the_top_k():
    assert recall_at_k([8, 2, 4, 6], [2, 2, 4, 6, 9], k=3) == 0.5
    assert recall_at_k(range(11), {10}) == 0.0


def test_ndcg_rewards_higher_ranks_and_normalizes_at_the_cutoff():
    assert ndcg_at_k([2, 4, 8], {2, 4, 6}, k=2) == 1.0
    assert ndcg_at_k([8, 2, 4], {2, 4, 6}, k=2) == pytest.approx((1 / log2(3)) / (1 + 1 / log2(3)))
    assert ndcg_at_k([2], {2, 4}, k=2) == pytest.approx(1 / (1 + 1 / log2(3)))


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_accuracy_is_zero_without_hits_or_positives(metric):
    assert metric([], {1}) == 0.0
    assert metric([1], []) == 0.0
    assert metric([1], {2}) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_accuracy_checks_cutoff_and_duplicate_recommendations(metric):
    with pytest.raises(ValueError, match="positive"):
        metric([1], {1}, k=0)
    with pytest.raises(ValueError, match="duplicate"):
        metric([1, 1], {1})
    assert metric([1, 1], {1}, k=1) == 1.0


def test_novelty_and_exposure_use_the_data_notebook_columns():
    stats = pl.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "novelty": [2.0, 4.0, 6.0, 8.0, None],
            "is_long_tail": [False, True, True, True, True],
            "recommendable": [True, True, True, True, False],
        }
    )
    novelty = dict(stats.select("movieId", "novelty").iter_rows())
    long_tail = set(stats.filter("is_long_tail")["movieId"])
    catalog = stats.filter("recommendable")["movieId"]
    rankings = [[1, 2, 4], [2, 3, 5]]

    assert novelty_at_k(rankings[0], novelty, k=2) == 3.0
    assert long_tail_share_at_k(rankings[0], long_tail, k=2) == 0.5
    assert catalog_coverage(rankings, catalog, k=2) == 0.75
    assert novelty_at_k([1, 3], novelty) == 4.0
    assert long_tail_share_at_k([2], long_tail) == 1.0


def test_empty_rankings_and_catalog():
    assert catalog_coverage([], [1, 2]) == 0.0
    assert catalog_coverage([[]], []) == 0.0
    assert novelty_at_k([], {}) == 0.0
    assert long_tail_share_at_k([], set()) == 0.0
    with pytest.raises(ValueError, match="positive"):
        catalog_coverage([], [], k=0)


def test_catalog_coverage_rejects_ineligible_movies():
    with pytest.raises(ValueError, match="outside the catalog"):
        catalog_coverage([[2]], [1])


@pytest.mark.parametrize("novelty", [None, float("nan"), float("inf")])
def test_novelty_rejects_undefined_values(novelty):
    with pytest.raises(ValueError, match="no finite novelty"):
        novelty_at_k([1], {1: novelty})


def test_genre_diversity_averages_unordered_pairs():
    genres = {1: ["Drama", "Comedy"], 2: ["Drama"], 3: ["Western"]}
    assert genre_diversity_at_k([1, 2, 3], genres) == pytest.approx((0.5 + 1 + 1) / 3)
    assert genre_diversity_at_k([1, 2, 3], genres, k=2) == 0.5
    assert genre_diversity_at_k([1, 2], {1: [], 2: []}) == 0.0
    assert genre_diversity_at_k([1], {}) == 0.0


def test_embedding_diversity_uses_cosine_distance():
    embeddings = {1: [1, 0], 2: [5, 0], 3: [0, 4], 4: [-7, 0]}
    assert embedding_diversity_at_k([1, 2, 3], embeddings) == pytest.approx(2 / 3)
    assert embedding_diversity_at_k([1, 2, 3], embeddings, k=2) == 0.0
    assert embedding_diversity_at_k([1, 4], embeddings) == 2.0
    assert embedding_diversity_at_k([], {}) == 0.0


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan"), 1], [[1, 0]]])
def test_embedding_diversity_rejects_invalid_vectors(vector):
    with pytest.raises(ValueError, match="nonzero embedding"):
        embedding_diversity_at_k([1, 2], {1: [1, 0], 2: vector})


def test_diversity_requires_features_for_every_compared_movie():
    with pytest.raises(KeyError):
        genre_diversity_at_k([1, 2], {1: ["Drama"]})
    with pytest.raises(KeyError):
        embedding_diversity_at_k([1, 2], {1: [1, 0]})
    with pytest.raises(ValueError):
        embedding_diversity_at_k([1, 2], {1: [1, 0], 2: [1, 0, 1]})
