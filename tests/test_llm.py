import json

import pytest

from movierec.llm import (
    answer_labels,
    answer_schema,
    borda,
    candidate_labels,
    describe,
    mean_kendall_tau,
    messages,
    parse_ranking,
    top_overlap,
    user_message,
)


def test_describe_adds_genres_when_known():
    assert describe("Heat (1995)", ["Action", "Crime"]) == "Heat (1995) [Action, Crime]"
    assert describe("Unknown (2001)", None) == "Unknown (2001)"


def test_candidate_labels_and_schema():
    assert candidate_labels(3) == ["1", "2", "3"]
    assert candidate_labels(3, "letters") == ["A", "B", "C"]
    assert answer_schema("letters")["properties"]["ranking"]["items"] == {"type": "string"}
    with pytest.raises(ValueError):
        candidate_labels(27, "letters")


def test_user_message_numbers_candidates_from_one():
    text = user_message([("Heat (1995)", 4.5), ("Up (2009)", 3.0)], ["A (2000)", "B (2001)"])
    assert "- Heat (1995): 4.5\n- Up (2009): 3" in text
    assert "1. A (2000)\n2. B (2001)" in text
    assert "every number from 1 to 2" in text


def test_user_message_with_letters_and_a_top_list():
    text = user_message([("Heat (1995)", 4.0)], ["X (2000)", "Y (2001)", "Z (2002)"], "letters", 2)
    assert "A. X (2000)\nB. Y (2001)\nC. Z (2002)" in text
    assert "Choose the 2 of these 3" in text
    assert "listing 2 different letters" in text


def test_messages_put_examples_before_the_request():
    example = (user_message([("X (1990)", 5.0)], ["Y (1991)", "Z (1992)"]), [2, 1])
    chat = messages([("Heat (1995)", 4.0)], ["A (2000)"], examples=[example])
    assert [message["role"] for message in chat] == ["system", "user", "assistant", "user"]
    assert json.loads(chat[2]["content"]) == {"ranking": [2, 1]}
    assert "Heat (1995)" in chat[3]["content"]


def test_answer_labels_follow_the_style():
    assert answer_labels([3, 1]) == [3, 1]
    assert answer_labels([3, 1], "letters") == ["C", "A"]


def test_parse_ranking_reads_valid_json():
    assert parse_ranking('{"ranking": [3, 1, 2]}', 3) == ([3, 1, 2], 0, 0)


def test_parse_ranking_drops_bad_entries_and_appends_missing_ones():
    # 7 is out of range and the second 3 is a repeat; 2 and 4 are missing.
    assert parse_ranking('{"ranking": [3, 7, 1, 3]}', 4) == ([3, 1, 2, 4], 2, 2)


def test_parse_ranking_falls_back_to_the_numbers_in_plain_text():
    assert parse_ranking("Best: 2, then 1.", 3) == ([2, 1, 3], 0, 1)
    assert parse_ranking("no idea", 2) == ([1, 2], 0, 2)


def test_parse_ranking_with_letters_and_a_top_list():
    assert parse_ranking('{"ranking": ["c", "A", "Z"]}', 4, "letters") == ([3, 1, 2, 4], 1, 2)
    # Only the top 2 is kept; a short answer is filled in the shown order.
    assert parse_ranking('{"ranking": ["D", "B", "A"]}', 4, "letters", 2) == ([4, 2], 0, 0)
    assert parse_ranking('{"ranking": [3]}', 4, top_n=2) == ([3, 1], 0, 1)


def test_borda_ranks_by_mean_position_and_breaks_ties_by_id():
    assert borda([[10, 20, 30], [20, 10, 30], [10, 30, 20]]) == [10, 20, 30]
    assert borda([[2, 1], [1, 2]]) == [1, 2]


def test_borda_combines_top_lists_of_different_movies():
    # 30 is missing from the second list, so it counts as position 2 there.
    assert borda([[10, 30], [20, 10]]) == [10, 20, 30]


def test_borda_rejects_lists_of_different_length_or_with_repeats():
    with pytest.raises(ValueError):
        borda([[1, 2], [1]])
    with pytest.raises(ValueError):
        borda([[1, 1], [1, 2]])


def test_mean_kendall_tau_is_one_for_identical_and_minus_one_for_reversed():
    assert mean_kendall_tau([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) == pytest.approx(1.0)
    assert mean_kendall_tau([[1, 2, 3], [3, 2, 1]]) == pytest.approx(-1.0)


def test_top_overlap_counts_shared_movies_in_the_top_k():
    assert top_overlap([[1, 2, 3], [2, 1, 4]], k=2) == pytest.approx(1.0)
    assert top_overlap([[1, 2, 3], [3, 4, 1]], k=2) == pytest.approx(0.0)
    assert top_overlap([[1, 2], [1, 3], [1, 2]], k=2) == pytest.approx((1 + 0.5 + 0.5) / 3 / 1)
