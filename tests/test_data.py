import polars as pl
import pytest

from movierec.data import chronological_split, long_tail


def ratings_for(user_id, n):
    return pl.DataFrame(
        {
            "userId": [user_id] * n,
            "movieId": list(range(n)),
            "rating": [4.0] * n,
            "timestamp": list(range(100, 100 + n)),
        }
    )


def splits(frame, user_id):
    return frame.filter(pl.col("userId") == user_id).sort("timestamp")["split"].to_list()


def test_split_holds_out_the_most_recent_ratings():
    split = chronological_split(ratings_for(1, 20))
    # 20 ratings: 4 for testing, then 4 of the remaining 16 (rounded up) for validation.
    assert splits(split, 1) == ["train"] * 12 + ["val"] * 4 + ["test"] * 4


def test_split_rounds_up_so_every_user_has_a_test_rating():
    split = chronological_split(pl.concat([ratings_for(1, 1), ratings_for(2, 3)]))
    assert splits(split, 1) == ["test"]
    assert splits(split, 2) == ["train", "val", "test"]


def test_split_is_per_user_and_ignores_input_order():
    ratings = pl.concat([ratings_for(1, 10), ratings_for(2, 5)]).reverse()
    split = chronological_split(ratings)
    assert splits(split, 1) == ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    assert splits(split, 2) == ["train"] * 3 + ["val"] + ["test"]


def test_split_keeps_an_exact_share_as_it_is():
    split = chronological_split(ratings_for(1, 15))
    assert splits(split, 1).count("test") == 3


def test_split_breaks_timestamp_ties_by_movie_id():
    ratings = pl.DataFrame(
        {"userId": [1] * 5, "movieId": [5, 4, 3, 2, 1], "rating": [4.0] * 5, "timestamp": [7] * 5}
    )
    split = chronological_split(ratings, test_fraction=0.4, val_fraction=0.0)
    test = split.filter(pl.col("split") == "test")["movieId"].sort().to_list()
    assert test == [4, 5]


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        chronological_split(ratings_for(1, 5), test_fraction=0.0)


def test_long_tail_keeps_the_smallest_head_with_80_percent_of_ratings():
    counts = pl.DataFrame({"movieId": [1, 2, 3, 4, 5], "n_ratings": [50, 30, 10, 10, 0]})
    tail = dict(long_tail(counts).select("movieId", "is_long_tail").iter_rows())
    # Movies 1 and 2 hold exactly 80% of the ratings.
    assert tail == {1: False, 2: False, 3: True, 4: True, 5: True}


def test_long_tail_includes_the_movie_that_crosses_the_threshold():
    counts = pl.DataFrame({"movieId": [1, 2, 3], "n_ratings": [70, 20, 10]})
    tail = dict(long_tail(counts).select("movieId", "is_long_tail").iter_rows())
    assert tail == {1: False, 2: False, 3: True}
