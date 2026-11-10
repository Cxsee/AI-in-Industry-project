"""Splits and popularity statistics shared by all notebooks."""

import polars as pl


def chronological_split(
    ratings: pl.DataFrame, test_fraction: float = 0.2, val_fraction: float = 0.2
) -> pl.DataFrame:
    """Add a `split` column with "train", "val" or "test" to each rating.

    For every user, the most recent `test_fraction` of their ratings goes to "test". Of the
    ratings left, the most recent `val_fraction` goes to "val", and the rest to "train".
    Both shares are rounded up, so a user with at least one rating always has a test rating.
    Ratings with the same timestamp are ordered by `movieId`, so the split is deterministic.
    """
    if not 0 < test_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1) and val_fraction in [0, 1).")
    n = pl.len().over("userId")
    position = pl.int_range(pl.len()).over("userId")
    # Rounding first guards against float noise pushing an exact share just above an integer,
    # which would make ceil() add a rating.
    n_test = (n * test_fraction).round(9).ceil().cast(pl.Int64)
    n_rest = n - n_test
    n_val = (n_rest * val_fraction).round(9).ceil().cast(pl.Int64)
    return ratings.sort(["userId", "timestamp", "movieId"]).with_columns(
        pl.when(position >= n_rest)
        .then(pl.lit("test"))
        .when(position >= n_rest - n_val)
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        # An Enum stores each label once, which matters with 32 million ratings.
        .cast(pl.Enum(["train", "val", "test"]))
        .alias("split")
    )


def long_tail(counts: pl.DataFrame, head_share: float = 0.8) -> pl.DataFrame:
    """Add an `is_long_tail` column to a frame with `movieId` and `n_ratings`.

    The head is the smallest set of most-rated movies that together hold `head_share` of all
    ratings. Every other movie, including those with no ratings, is long tail. Ties in the
    number of ratings are broken by `movieId`, so the result is deterministic.
    """
    total = counts["n_ratings"].sum()
    if total == 0:
        return counts.with_columns(pl.lit(True).alias("is_long_tail"))
    ranked = counts.sort(["n_ratings", "movieId"], descending=[True, False])
    share_before = (ranked["n_ratings"].cum_sum() - ranked["n_ratings"]) / total
    return ranked.with_columns(
        ((share_before >= head_share) | (pl.col("n_ratings") == 0)).alias("is_long_tail")
    )
