import numpy as np
import polars as pl
import pytest

from movierec.interactions import interaction_matrix, mixed_candidates, rankings_frame


def test_interaction_matrix_is_binary_and_sorted():
    pairs = pl.DataFrame({"userId": [30, 10, 10, 30, 30], "movieId": [7, 5, 9, 5, 5]})
    matrix, user_ids, movie_ids = interaction_matrix(pairs)
    assert user_ids.tolist() == [10, 30]
    assert movie_ids.tolist() == [5, 7, 9]
    # The repeated (30, 5) pair stays a single 1.
    assert matrix.toarray().tolist() == [[1, 0, 1], [1, 1, 0]]
    assert matrix.dtype == np.float32


def test_rankings_frame_maps_columns_back_to_movie_ids():
    frame = rankings_frame(
        user_ids=np.array([10, 30]),
        columns=np.array([[2, 0], [1, -1]]),
        scores=np.array([[0.9, 0.5], [0.7, 0.0]]),
        movie_ids=np.array([5, 7, 9]),
    )
    assert frame.columns == ["userId", "movieId", "rank", "score"]
    assert frame.rows() == [(10, 9, 1, 0.9), (10, 5, 2, 0.5), (30, 7, 1, 0.7)]


def candidates(rows):
    return pl.DataFrame(rows, schema=["userId", "movieId", "rank", "score"], orient="row")


def test_mixed_candidates_ranks_by_score_and_keeps_the_origin():
    head = candidates([(1, 10, 1, 0.9), (1, 11, 2, 0.4), (2, 10, 1, 0.8)])
    tail = candidates([(1, 20, 1, 0.5), (1, 21, 2, 0.4), (2, 20, 1, 0.1)])
    merged = mixed_candidates(head, tail)
    assert merged.columns == ["userId", "movieId", "rank", "score", "is_long_tail"]
    # User 1: 0.9 head, then 0.5 tail, then a 0.4 tie broken by movieId.
    assert merged.rows() == [
        (1, 10, 1, 0.9, False),
        (1, 20, 2, 0.5, True),
        (1, 11, 3, 0.4, False),
        (1, 21, 4, 0.4, True),
        (2, 10, 1, 0.8, False),
        (2, 20, 2, 0.1, True),
    ]


def test_mixed_candidates_rejects_a_movie_in_both_lists():
    head = candidates([(1, 10, 1, 0.9)])
    tail = candidates([(1, 10, 1, 0.9)])
    with pytest.raises(ValueError, match="both"):
        mixed_candidates(head, tail)
