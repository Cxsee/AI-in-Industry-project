"""Content profiles from multilingual plots and temporally filtered metadata."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import polars as pl


def fitting_history(ratings: pl.LazyFrame, split: str = "val") -> pl.LazyFrame:
    """Match the fitting windows used by Popularity and MF."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test.")
    return ratings.filter(pl.col("split").is_in(["train"] if split == "val" else ["train", "val"]))


def movie_texts(
    movies: pl.DataFrame,
    plots: pl.DataFrame,
    tags: pl.LazyFrame,
    history: pl.LazyFrame,
    *,
    max_tags: int = 20,
) -> pl.DataFrame:
    """Use accepted plots, else genres and tags available in the author's fit window.

    A tag must strictly precede its author's last fitting rating. This conservative
    cutoff also excludes held-out events and ambiguous boundary timestamp ties.
    This is a per-author cutoff, consistent with the project's per-user split;
    it does not impose a global chronological cutoff across different users.
    """
    if max_tags < 1:
        raise ValueError("max_tags must be positive.")
    for frame, label in [(movies, "movies"), (plots, "plots")]:
        if frame["movieId"].null_count() or frame["movieId"].n_unique() != frame.height:
            raise ValueError(f"{label} must have unique non-null movieId values.")
    cutoffs = history.group_by("userId").agg(pl.col("timestamp").max().alias("cutoff"))
    # Use < rather than <= to exclude all ambiguous events at a boundary timestamp.
    available = (
        tags.join(cutoffs, on="userId", how="inner")
        .filter(pl.col("timestamp") < pl.col("cutoff"))
        .join(movies.select("movieId").lazy(), on="movieId", how="semi")
        .with_columns(pl.col("tag").str.strip_chars().str.to_lowercase())
        .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
        .group_by("movieId", "tag")
        .len()
        .sort(["movieId", "len", "tag"], descending=[False, True, False])
        .group_by("movieId", maintain_order=True)
        .agg(pl.col("tag").head(max_tags))
        .collect()
    )
    tag_map = dict(available.iter_rows())
    plot_map = {
        row["movieId"]: row
        for row in plots.iter_rows(named=True)
        if row["status"] == "ok" and row["plot"] and row["plot"].strip()
    }
    rows = []
    for movie in movies.sort("movieId").iter_rows(named=True):
        mid = movie["movieId"]
        plot = plot_map.get(mid)
        genres = [g for g in movie["genres"] or [] if g != "(no genres listed)"]
        movie_tags = tag_map.get(mid, [])
        if plot:
            text, source = plot["plot"].strip(), "plot"
        else:
            parts = []
            if genres:
                parts.append("Genres: " + ", ".join(genres))
            if movie_tags:
                parts.append("Tags: " + ", ".join(movie_tags))
            text = ". ".join(parts)
            source = "genres_tags" if parts else "missing"
        rows.append((mid, text, source, plot.get("plot_language") if plot else None))
    return pl.DataFrame(
        rows,
        schema={
            "movieId": movies["movieId"].dtype,
            "text": pl.String,
            "text_source": pl.String,
            "plot_language": pl.String,
        },
        orient="row",
    )


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encode_movie_texts(
    texts: pl.DataFrame,
    model,
    cache_dir: Path,
    *,
    model_id: str,
    revision: str,
    batch_size: int = 32,
    checkpoint_movies: int = 256,
) -> np.ndarray:
    """Encode all plot chunks; cache shards keyed by ordered IDs, texts and encoder.

    Chunk vectors are averaged by token count, then L2-normalized. Empty metadata
    gets a zero vector and never creates an arbitrary semantic recommendation.
    Only importing the notebook's optional extra loads Sentence Transformers.
    """
    if batch_size < 1 or checkpoint_movies < 1 or not revision:
        raise ValueError("Positive batch sizes and an explicit model revision are required.")
    budget = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1:
        raise ValueError("The encoder has no space for text tokens.")
    spec = [model_id, revision, model.max_seq_length, "token-weighted-chunks-v1", texts.rows()]
    fingerprint = hashlib.sha256(json.dumps(spec, ensure_ascii=False).encode()).hexdigest()
    root = Path(cache_dir) / fingerprint
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    output = np.zeros((texts.height, dim), dtype=np.float32)
    for offset in range(0, texts.height, checkpoint_movies):
        shard = texts.slice(offset, checkpoint_movies)
        ids = shard["movieId"].to_numpy()
        path = root / f"{offset:07d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                vectors = cached["embeddings"]
                if not np.array_equal(cached["movie_ids"], ids) or vectors.shape != (len(ids), dim):
                    raise ValueError(f"Invalid embedding checkpoint: {path}")
        else:
            chunks, owners, weights = [], [], []
            for i, text in enumerate(shard["text"]):
                tokens = (
                    model.tokenizer.encode(text, add_special_tokens=False, verbose=False)
                    if text
                    else []
                )
                for start in range(0, len(tokens), budget):
                    part = tokens[start : start + budget]
                    chunks.append(model.tokenizer.decode(part, skip_special_tokens=True))
                    owners.append(i)
                    weights.append(len(part))
            vectors = np.zeros((len(ids), dim), dtype=np.float32)
            if chunks:
                encoded = model.encode(
                    chunks,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                np.add.at(vectors, owners, np.asarray(encoded) * np.asarray(weights)[:, None])
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors /= np.where(norms > 0, norms, 1)
            _atomic_npz(path, movie_ids=ids, embeddings=vectors)
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must be finite.")
        output[offset : offset + len(ids)] = vectors
    return output


def recommend_content(
    history: pl.DataFrame,
    users: pl.DataFrame,
    movie_ids: np.ndarray,
    embeddings: np.ndarray,
    catalog: pl.DataFrame,
    *,
    k: int = 10,
    relevant_rating: float = 4.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mean liked-film profile, cosine ranking, all seen films excluded.

    For a user without a usable positive profile, use train-only popularity and
    record the fallback. Ties always use ascending movieId. Score means cosine
    for semantic rows and rating count for popularity fallback rows.
    """
    movie_ids = np.asarray(movie_ids)
    vectors = np.array(embeddings, dtype=np.float32, copy=True)
    if k < 1 or vectors.ndim != 2 or vectors.shape[0] != len(movie_ids):
        raise ValueError("Invalid k or embedding shape.")
    if len(set(movie_ids)) != len(movie_ids) or not np.isfinite(vectors).all():
        raise ValueError("Movie IDs must be unique and embeddings finite.")
    if users["userId"].null_count() or users["userId"].n_unique() != users.height:
        raise ValueError("Users must have unique non-null IDs.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.where(norms > 0, norms, 1)
    indices = {mid: i for i, mid in enumerate(movie_ids)}
    catalog = catalog.sort("movieId")
    candidate_ids = catalog["movieId"].to_numpy()
    if set(candidate_ids) - indices.keys():
        raise ValueError("Every candidate must have an embedding row, including missing metadata.")
    candidate_rows = np.array([indices[mid] for mid in candidate_ids], dtype=np.int64)
    candidate_vectors = vectors[candidate_rows]
    usable = np.linalg.norm(candidate_vectors, axis=1) > 0
    counts = catalog["n_ratings"].to_numpy()
    histories = {
        key[0]: group for key, group in history.partition_by("userId", as_dict=True).items()
    }
    rows, diagnostics = [], []
    for user in users["userId"]:
        past = histories.get(user)
        seen = set(past["movieId"]) if past is not None else set()
        liked = (
            past.filter(pl.col("rating") >= relevant_rating)["movieId"] if past is not None else []
        )
        liked_rows = [indices[mid] for mid in liked if mid in indices]
        liked_rows = [i for i in liked_rows if np.linalg.norm(vectors[i]) > 0]
        profile = vectors[liked_rows].mean(axis=0) if liked_rows else np.zeros(vectors.shape[1])
        norm = np.linalg.norm(profile)
        fallback = norm <= 1e-8
        scores = counts.astype(float) if fallback else candidate_vectors @ (profile / norm)
        eligible = np.array([mid not in seen for mid in candidate_ids], dtype=bool)
        if not fallback:
            eligible &= usable
        positions = np.flatnonzero(eligible)
        order = positions[np.lexsort((candidate_ids[positions], -scores[positions]))][:k]
        rows.extend(
            (user, int(candidate_ids[i]), rank, float(scores[i])) for rank, i in enumerate(order, 1)
        )
        diagnostics.append((user, len(liked_rows), fallback, len(order)))
    rankings = pl.DataFrame(
        rows,
        schema={
            "userId": users["userId"].dtype,
            "movieId": pl.Int64,
            "rank": pl.Int64,
            "score": pl.Float64,
        },
        orient="row",
    )
    diagnostic = pl.DataFrame(
        diagnostics,
        schema={
            "userId": users["userId"].dtype,
            "n_profile_movies": pl.Int64,
            "popularity_fallback": pl.Boolean,
            "n_recommended": pl.Int64,
        },
        orient="row",
    )
    return rankings, diagnostic
