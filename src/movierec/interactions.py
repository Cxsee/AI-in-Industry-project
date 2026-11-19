"""Turn ratings into the user × movie matrices that factorization models train on."""

import numpy as np
import polars as pl
from scipy import sparse


def interaction_matrix(
    pairs: pl.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build a binary user × movie matrix from a frame with `userId` and `movieId`.

    Returns the matrix with the user IDs of its rows and the movie IDs of its columns, both
    sorted. Every pair present is a 1, whatever the rating, and repeated pairs stay 1.
    """
    users = pairs["userId"].to_numpy()
    movies = pairs["movieId"].to_numpy()
    user_ids = np.unique(users)
    movie_ids = np.unique(movies)
    matrix = sparse.csr_matrix(
        (
            np.ones(len(pairs), dtype=np.float32),
            (np.searchsorted(user_ids, users), np.searchsorted(movie_ids, movies)),
        ),
        shape=(len(user_ids), len(movie_ids)),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1
    return matrix, user_ids, movie_ids


def rankings_frame(
    user_ids: np.ndarray, columns: np.ndarray, scores: np.ndarray, movie_ids: np.ndarray
) -> pl.DataFrame:
    """Turn per-user top-N columns and scores into `userId`, `movieId`, `rank`, `score` rows.

    `columns` and `scores` have one row per user, best first, as returned by the
    `recommend` method of the `implicit` models. A column of -1 marks a missing item and
    is dropped, so a user can get a shorter list.
    """
    columns = np.asarray(columns)
    scores = np.asarray(scores)
    n_users, n_items = columns.shape
    frame = pl.DataFrame(
        {
            "userId": np.repeat(np.asarray(user_ids), n_items),
            "column": columns.ravel(),
            "rank": np.tile(np.arange(1, n_items + 1, dtype=np.int64), n_users),
            "score": scores.ravel().astype(np.float64),
        }
    ).filter(pl.col("column") >= 0)
    return frame.with_columns(
        pl.Series("movieId", np.asarray(movie_ids)[frame["column"].to_numpy()])
    ).select("userId", "movieId", "rank", "score")


def mixed_candidates(head: pl.DataFrame, tail: pl.DataFrame) -> pl.DataFrame:
    """Merge each user's head and long-tail candidates into one list.

    Both frames have `userId`, `movieId`, `rank` and `score`, from the same model, and no
    movie may appear in both for the same user. The result is ranked again from 1 by score,
    best first, with ties broken by `movieId`, and keeps the origin in `is_long_tail`.
    """
    merged = pl.concat(
        [
            head.with_columns(pl.lit(False).alias("is_long_tail")),
            tail.with_columns(pl.lit(True).alias("is_long_tail")),
        ]
    )
    if merged.select("userId", "movieId").is_duplicated().any():
        raise ValueError("A movie is both a head and a long-tail candidate for the same user.")
    return (
        merged.sort(["userId", "score", "movieId"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("userId").alias("rank"))
        .select("userId", "movieId", "rank", "score", "is_long_tail")
    )
