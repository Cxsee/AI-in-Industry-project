"""Offline metrics for ordered MovieLens recommendations.

Pass movie IDs in rank order and provide the relevance judgments and catalog for
the split being evaluated. The default cutoff is 10. Short lists use their actual
length for averages and shares, and duplicate IDs in the evaluated prefix are
rejected. An empty prefix scores zero.
"""

from collections.abc import Iterable, Mapping
from itertools import combinations, islice
from math import isfinite, log2
from statistics import mean

import numpy as np
from numpy.typing import ArrayLike


def _top_k(recommended: Iterable[int], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive.")
    top = list(islice(recommended, k))
    if len(top) != len(set(top)):
        raise ValueError("The top-k recommendations contain duplicate movie IDs.")
    return top


def recall_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> float:
    """Return the share of distinct relevant movies retrieved, or zero without positives."""
    top = _top_k(recommended, k)
    positives = set(relevant)
    return len(positives.intersection(top)) / len(positives) if positives else 0.0


def ndcg_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> float:
    """Return binary NDCG using the ideal ranking of up to k relevant movies.

    A short recommendation list can miss relevant items that fit in the ideal
    top-k. A user without relevant movies scores zero.
    """
    top = _top_k(recommended, k)
    positives = set(relevant)
    if not top or not positives:
        return 0.0
    dcg = sum(1 / log2(rank + 1) for rank, movie in enumerate(top, 1) if movie in positives)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(positives)) + 1))
    return dcg / ideal_dcg


def novelty_at_k(
    recommended: Iterable[int], novelty_by_movie: Mapping[int, float | None], k: int = 10
) -> float:
    """Average the precomputed `novelty` column from movie_stats.parquet.

    Every recommended movie must have a finite novelty value.
    """
    values = []
    for movie in _top_k(recommended, k):
        novelty = novelty_by_movie[movie]
        if novelty is None or not isfinite(novelty):
            raise ValueError(f"Movie {movie} has no finite novelty.")
        values.append(novelty)
    return mean(values) if values else 0.0


def catalog_coverage(
    rankings: Iterable[Iterable[int]], catalog: Iterable[int], k: int = 10
) -> float:
    """Return the fraction of distinct eligible movies recommended across all top-k lists."""
    if k <= 0:
        raise ValueError("k must be positive.")
    eligible = set(catalog)
    exposed = {movie for ranking in rankings for movie in _top_k(ranking, k)}
    outside_catalog = exposed - eligible
    if outside_catalog:
        raise ValueError(f"Recommended movie {min(outside_catalog)} is outside the catalog.")
    return len(exposed) / len(eligible) if eligible else 0.0


def long_tail_share_at_k(
    recommended: Iterable[int], long_tail_movies: Iterable[int], k: int = 10
) -> float:
    """Return the fraction of recommended movies marked `is_long_tail`."""
    top = _top_k(recommended, k)
    return len(set(top).intersection(long_tail_movies)) / len(top) if top else 0.0


def genre_diversity_at_k(
    recommended: Iterable[int], genres_by_movie: Mapping[int, Iterable[str]], k: int = 10
) -> float:
    """Return mean pairwise Jaccard distance between genre sets.

    Two empty genre sets have distance zero. Fewer than two recommendations
    score zero. Supply a genre collection for every movie in a compared pair.
    """
    top = _top_k(recommended, k)
    if len(top) < 2:
        return 0.0
    genres = [set(genres_by_movie[movie]) for movie in top]
    return mean(
        len(first ^ second) / len(first | second) if first or second else 0.0
        for first, second in combinations(genres, 2)
    )


def embedding_diversity_at_k(
    recommended: Iterable[int], embeddings_by_movie: Mapping[int, ArrayLike], k: int = 10
) -> float:
    """Return mean pairwise cosine distance in [0, 2].

    Fewer than two recommendations score zero. Each compared movie needs a
    finite, nonzero vector of the same dimension.
    """
    top = _top_k(recommended, k)
    if len(top) < 2:
        return 0.0

    vectors = []
    for movie in top:
        vector = np.asarray(embeddings_by_movie[movie], dtype=float)
        if vector.ndim != 1 or not np.isfinite(vector).all() or not vector.any():
            raise ValueError(f"Movie {movie} needs a finite, nonzero embedding vector.")
        scaled = vector / np.abs(vector).max()
        vectors.append(scaled / np.linalg.norm(scaled))

    unit_vectors = np.stack(vectors)
    rows, columns = np.triu_indices(len(top), k=1)
    similarities = (unit_vectors @ unit_vectors.T)[rows, columns]
    return float(np.mean(1 - np.clip(similarities, -1, 1)))
