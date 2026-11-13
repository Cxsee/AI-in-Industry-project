"""Evaluate recommendation rankings against the common MovieLens validation split."""

import polars as pl

from movierec.data import long_tail
from movierec.metrics import (
    catalog_coverage,
    genre_diversity_at_k,
    long_tail_share_at_k,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)


def evaluate_validation(
    ratings: pl.LazyFrame,
    users: pl.DataFrame,
    movies: pl.DataFrame,
    recommendations: pl.DataFrame,
    *,
    k: int = 10,
    min_ratings: int = 10,
    relevant_rating: float = 4.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return per-user metrics and a one-row summary, using train and val ratings only.

    Users without a validation positive have null accuracy scores and still count
    toward beyond-accuracy metrics. An absent ranking is an empty list. Every
    validation positive counts toward the ideal ranking and recall denominator.
    """
    if k < 1 or min_ratings < 1:
        raise ValueError("k and min_ratings must be positive.")

    user_ids = users["userId"]
    if user_ids.null_count() or user_ids.n_unique() != len(user_ids):
        raise ValueError("The evaluation sample must contain distinct, non-null user IDs.")

    train = ratings.filter(pl.col("split") == "train").select("userId", "movieId")
    sample = users.select("userId").lazy()
    counts, train_users, seen_rows, positive_rows = pl.collect_all(
        [
            train.group_by("movieId").len(name="n_ratings"),
            train.select(pl.col("userId").n_unique()),
            train.join(sample, on="userId", how="semi").group_by("userId").agg("movieId"),
            ratings.filter((pl.col("split") == "val") & (pl.col("rating") >= relevant_rating))
            .join(sample, on="userId", how="semi")
            .group_by("userId")
            .agg("movieId"),
        ]
    )
    stats = long_tail(counts).filter(pl.col("n_ratings") >= min_ratings)
    catalog = set(stats["movieId"])
    novelty = dict(
        stats.select(
            "movieId", -(pl.col("n_ratings") / max(1, train_users.item())).log(2)
        ).iter_rows()
    )
    tail = set(stats.filter("is_long_tail")["movieId"])
    genres = {
        movie: values or [] for movie, values in movies.select("movieId", "genres").iter_rows()
    }
    seen = {user: set(items) for user, items in seen_rows.iter_rows()}
    positives = dict(positive_rows.iter_rows())
    rankings = _validated_rankings(recommendations, set(user_ids), catalog, seen, k)

    rows = []
    for user in user_ids:
        ranked = rankings.get(user, [])
        relevant = positives.get(user, [])
        rows.append(
            (
                user,
                len(ranked),
                len(relevant),
                recall_at_k(ranked, relevant, k) if relevant else None,
                ndcg_at_k(ranked, relevant, k) if relevant else None,
                novelty_at_k(ranked, novelty, k),
                genre_diversity_at_k(ranked, genres, k),
                long_tail_share_at_k(ranked, tail, k),
            )
        )

    metric_columns = ["recall", "ndcg", "novelty", "genre_diversity", "long_tail_share"]
    per_user = pl.DataFrame(
        rows,
        schema={
            "userId": user_ids.dtype,
            "n_recommended": pl.Int64,
            "n_positives": pl.Int64,
            **dict.fromkeys(metric_columns, pl.Float64),
        },
        orient="row",
    )
    summary = per_user.select(
        pl.len().alias("n_users"),
        (pl.col("n_positives") > 0).sum().alias("n_accuracy_users"),
        (pl.col("n_positives") == 0).sum().alias("n_excluded_no_positives"),
        (pl.col("n_recommended") == 0).sum().alias("n_empty_rankings"),
        pl.col(metric_columns).mean(),
    ).with_columns(
        pl.lit(k).alias("k"),
        pl.lit(min_ratings).alias("min_ratings"),
        pl.lit(relevant_rating).alias("relevant_rating"),
        pl.lit(len(catalog)).alias("catalog_size"),
        pl.lit(catalog_coverage(rankings.values(), catalog, k)).alias("catalog_coverage"),
    )
    return per_user, summary


def _validated_rankings(
    recommendations: pl.DataFrame,
    users: set[int],
    catalog: set[int],
    seen: dict[int, set[int]],
    k: int,
) -> dict[int, list[int]]:
    ranked = recommendations.select("userId", "movieId", "rank")
    for name, dtype in ranked.schema.items():
        if not dtype.is_integer() or ranked[name].null_count():
            raise ValueError(f"Recommendation column {name} must contain non-null integers.")
    by_user = ranked.sort("rank").group_by("userId").agg("movieId", "rank")
    rankings = {}
    for user, movies, ranks in by_user.iter_rows():
        if user not in users:
            raise ValueError(f"User {user} is outside the evaluation sample.")
        if len(movies) > k or ranks != list(range(1, len(movies) + 1)):
            raise ValueError(f"User {user} needs consecutive ranks from 1 with at most {k} items.")
        movie_ids = set(movies)
        if len(movie_ids) != len(movies):
            raise ValueError(f"User {user} has duplicate recommended movies.")
        if movie_ids - catalog:
            raise ValueError(f"User {user} has recommendations outside the eligible catalog.")
        if movie_ids.intersection(seen.get(user, ())):
            raise ValueError(f"User {user} has recommendations already rated in train.")
        rankings[user] = movies
    return rankings
