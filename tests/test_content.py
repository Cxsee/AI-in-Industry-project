from pathlib import Path

import numpy as np
import polars as pl
import pytest

from movierec.content import (
    encode_movie_texts,
    fitting_history,
    movie_texts,
    recommend_content,
)


def test_fitting_window_and_tag_cutoffs():
    ratings = pl.DataFrame(
        {
            "userId": [1, 1, 1, 2],
            "movieId": [10, 20, 30, 10],
            "rating": [5.0, 5.0, 1.0, 5.0],
            "timestamp": [10, 20, 30, 10],
            "split": ["train", "val", "test", "train"],
        }
    ).lazy()
    movies = pl.DataFrame({"movieId": [10, 20, 30], "genres": [["Drama"], [], []]})
    plots = pl.DataFrame(
        {
            "movieId": [10, 20],
            "status": ["ok", "retryable_error"],
            "plot": ["A verified narrative", "Rejected text"],
            "plot_language": ["it", "en"],
        }
    )
    tags = pl.DataFrame(
        {
            "userId": [1, 1, 1, 1, 9],
            "movieId": [20] * 5,
            "timestamp": [9, 10, 19, 29, 1],
            "tag": [" EARLY ", "boundary", "validation", "test", "unknown"],
        }
    ).lazy()
    train = fitting_history(ratings)
    assert train.collect().height == 2
    val = movie_texts(movies, plots, tags, train)
    assert val["text"].to_list() == ["A verified narrative", "Tags: early", ""]
    assert val["text_source"].to_list() == ["plot", "genres_tags", "missing"]
    test = movie_texts(movies, plots, tags, fitting_history(ratings, "test"))
    assert test["text"][1] == "Tags: boundary, early, validation"
    with pytest.raises(ValueError, match="val or test"):
        fitting_history(ratings, "train")


def test_cosine_profiles_seen_exclusion_and_popularity_fallback():
    history = pl.DataFrame(
        {"userId": [1, 1, 2], "movieId": [10, 20, 10], "rating": [5.0, 1.0, 1.0]}
    )
    users = pl.DataFrame({"userId": [1, 2, 3]})
    ids = np.array([10, 20, 30, 40, 50])
    # Film 20 is disliked: it must neither enter the profile nor be recommended.
    embeddings = np.array([[2, 0], [0, 1], [1, 0], [1, 0], [0, 0]])
    catalog = pl.DataFrame({"movieId": [50, 40, 30, 20, 10], "n_ratings": [100, 20, 20, 30, 40]})
    ranked, diagnostics = recommend_content(history, users, ids, embeddings, catalog, k=3)
    one = ranked.filter(pl.col("userId") == 1)
    assert one["movieId"].to_list() == [30, 40]
    np.testing.assert_allclose(one["score"].to_numpy(), [1.0, 1.0])
    assert ranked.filter(pl.col("userId") == 2)["movieId"].to_list() == [50, 20, 30]
    assert diagnostics["popularity_fallback"].to_list() == [False, True, True]
    assert ranked.filter(pl.col("userId") == 3)["rank"].to_list() == [1, 2, 3]


def test_content_rankings_follow_evaluation_protocol():
    from movierec.evaluation import evaluate_validation

    ratings = pl.DataFrame(
        {
            "userId": [1, 2, 2, 1, 1],
            "movieId": [10, 20, 30, 20, 30],
            "rating": [5.0, 5.0, 4.0, 5.0, 1.0],
            "split": ["train", "train", "train", "val", "test"],
        }
    )
    users = pl.DataFrame({"userId": [1]})
    movies = pl.DataFrame({"movieId": [10, 20, 30], "genres": [["Drama"]] * 3})
    history = fitting_history(ratings.lazy()).collect()
    catalog = history.group_by("movieId").len(name="n_ratings")
    ranked, _ = recommend_content(
        history,
        users,
        np.array([10, 20, 30]),
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        catalog,
        k=2,
    )
    _, summary = evaluate_validation(ratings.lazy(), users, movies, ranked, k=2, min_ratings=1)
    assert summary["recall"].item() == 1.0
    assert summary["ndcg"].item() == 1.0
    # Held-out interactions cannot affect the fitted profile.
    altered = ratings.with_columns(
        pl.when(pl.col("split") != "train").then(0.0).otherwise(pl.col("rating")).alias("rating")
    )
    again, _ = recommend_content(
        fitting_history(altered.lazy()).collect(),
        users,
        np.array([10, 20, 30]),
        np.array([[1, 0], [1, 0], [0, 1]]),
        catalog,
        k=2,
    )
    assert again.equals(ranked)


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [int(word) for word in text.split()]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))


class FakeEncoder:
    max_seq_length = 4
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, chunks, **kwargs):
        self.calls.extend(chunks)
        return np.array([[float(chunk.split()[0]), 1.0] for chunk in chunks])


def test_embedding_chunks_checkpoint_resume_and_invalidation(tmp_path: Path):
    texts = pl.DataFrame({"movieId": [10, 20], "text": ["1 2 3 4 5", ""]})
    model = FakeEncoder()
    vectors = encode_movie_texts(
        texts, model, tmp_path, model_id="fake", revision="v1", checkpoint_movies=1
    )
    assert model.calls == ["1 2", "3 4", "5"]
    expected = np.array([2 * 1 + 2 * 3 + 1 * 5, 5.0])
    np.testing.assert_allclose(vectors[0], expected / np.linalg.norm(expected), rtol=1e-6)
    assert not vectors[1].any()
    again = encode_movie_texts(
        texts, model, tmp_path, model_id="fake", revision="v1", checkpoint_movies=1
    )
    assert len(model.calls) == 3
    np.testing.assert_array_equal(again, vectors)
    encode_movie_texts(texts, model, tmp_path, model_id="fake", revision="v2")
    assert len(model.calls) == 6
    changed = texts.with_columns(pl.lit("2").alias("text"))
    encode_movie_texts(changed, model, tmp_path, model_id="fake", revision="v1")
    assert len(model.calls) == 8


def test_missing_embedding_and_nonfinite_vectors_fail():
    history = pl.DataFrame(schema={"userId": pl.Int64, "movieId": pl.Int64, "rating": pl.Float64})
    users = pl.DataFrame({"userId": [1]})
    catalog = pl.DataFrame({"movieId": [2], "n_ratings": [10]})
    with pytest.raises(ValueError, match="Every candidate"):
        recommend_content(history, users, np.array([1]), np.ones((1, 2)), catalog)
    with pytest.raises(ValueError, match="finite"):
        recommend_content(history, users, np.array([2]), np.array([[np.nan, 0]]), catalog)
