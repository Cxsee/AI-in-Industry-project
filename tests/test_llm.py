import json

import pytest

from movierec.llm import (
    borda,
    describe,
    mean_kendall_tau,
    messages,
    parse_ranking,
    user_message,
)


def test_describe_adds_genres_when_known():
    assert describe("Heat (1995)", ["Action", "Crime"]) == "Heat (1995) [Action, Crime]"
    assert describe("Unknown (2001)", None) == "Unknown (2001)"


def test_user_message_numbers_candidates_from_one():
    text = user_message([("Heat (1995)", 4.5), ("Up (2009)", 3.0)], ["A (2000)", "B (2001)"])
    assert "- Heat (1995): 4.5\n- Up (2009): 3" in text
    assert "1. A (2000)\n2. B (2001)" in text
    assert "every number from 1 to 2" in text


def test_messages_put_examples_before_the_request():
    example = (user_message([("X (1990)", 5.0)], ["Y (1991)", "Z (1992)"]), [2, 1])
    chat = messages([("Heat (1995)", 4.0)], ["A (2000)"], examples=[example])
    assert [message["role"] for message in chat] == ["system", "user", "assistant", "user"]
    assert json.loads(chat[2]["content"]) == {"ranking": [2, 1]}
    assert "Heat (1995)" in chat[3]["content"]


def test_parse_ranking_reads_valid_json():
    assert parse_ranking('{"ranking": [3, 1, 2]}', 3) == ([3, 1, 2], 0, 0)


def test_parse_ranking_drops_bad_entries_and_appends_missing_ones():
    # 7 is out of range and the second 3 is a repeat; 2 and 4 are missing.
    assert parse_ranking('{"ranking": [3, 7, 1, 3]}', 4) == ([3, 1, 2, 4], 2, 2)


def test_parse_ranking_falls_back_to_the_numbers_in_plain_text():
    assert parse_ranking("Best: 2, then 1.", 3) == ([2, 1, 3], 0, 1)
    assert parse_ranking("no idea", 2) == ([1, 2], 0, 2)


def test_borda_ranks_by_mean_position_and_breaks_ties_by_id():
    assert borda([[10, 20, 30], [20, 10, 30], [10, 30, 20]]) == [10, 20, 30]
    assert borda([[2, 1], [1, 2]]) == [1, 2]


def test_borda_rejects_rankings_of_different_movies():
    with pytest.raises(ValueError):
        borda([[1, 2], [1, 3]])


def test_mean_kendall_tau_is_one_for_identical_and_minus_one_for_reversed():
    assert mean_kendall_tau([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) == pytest.approx(1.0)
    assert mean_kendall_tau([[1, 2, 3], [3, 2, 1]]) == pytest.approx(-1.0)
