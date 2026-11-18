import numpy as np
import polars as pl

from movierec.interactions import interaction_matrix, rankings_frame


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
