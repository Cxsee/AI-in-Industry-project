"""Prompts, answer parsing and rank aggregation for the LLM ranker.

Nothing here calls a model, so every piece can be tested without a GPU.

Candidates are shown with labels, numbers ("1", "2", ...) or letters ("A", "B", ...), and the
model answers with labels. All the functions below work with positions in the shown list,
counted from 1, so the rest of the code does not depend on the labels.
"""

import json
import re
from collections.abc import Sequence
from itertools import combinations
from string import ascii_uppercase

import numpy as np
from scipy.stats import kendalltau

SYSTEM_PROMPT = (
    "You are a movie recommender. You rank candidate movies by how likely a user is to "
    "enjoy them, judging only from the movies they rated. Answer with JSON only."
)
LABEL_STYLES = ("numbers", "letters")


def candidate_labels(n: int, style: str = "numbers") -> list[str]:
    """Labels of n candidates: "1".."n", or "A".. for letters (at most 26)."""
    if style == "numbers":
        return [str(number) for number in range(1, n + 1)]
    if style == "letters":
        if n > len(ascii_uppercase):
            raise ValueError("Letters can label at most 26 candidates.")
        return list(ascii_uppercase[:n])
    raise ValueError(f"style must be one of {LABEL_STYLES}.")


def answer_schema(style: str = "numbers") -> dict:
    """JSON schema for Ollama's `format`, which keeps the answer parseable."""
    item = {"type": "integer"} if style == "numbers" else {"type": "string"}
    return {
        "type": "object",
        "properties": {"ranking": {"type": "array", "items": item}},
        "required": ["ranking"],
    }


def describe(title: str, genres: Sequence[str] | None) -> str:
    """One line per movie: the MovieLens title, which ends with the year, and its genres."""
    return f"{title} [{', '.join(genres)}]" if genres else title


def user_message(
    history: Sequence[tuple[str, float]],
    candidates: Sequence[str],
    style: str = "numbers",
    top_n: int | None = None,
) -> str:
    """Ask to rank `candidates`, given the rated `history`, most recent first.

    `history` holds (description, rating) pairs. With `top_n`, only the best `top_n`
    candidates are asked for.
    """
    labels = candidate_labels(len(candidates), style)
    rated = "\n".join(f"- {movie}: {rating:g}" for movie, rating in history)
    listed = "\n".join(f"{label}. {movie}" for label, movie in zip(labels, candidates, strict=True))
    n = len(candidates)
    kind = "number" if style == "numbers" else "letter"
    if top_n is None:
        task = (
            f"Rank these {n} candidate movies "
            "from most to least likely to be enjoyed by the user"
        )
        answer = f"listing every {kind} from {labels[0]} to {labels[-1]} exactly once, best first"
    else:
        task = f"Choose the {top_n} of these {n} candidate movies the user is most likely to enjoy"
        answer = f"listing {top_n} different {kind}s, best first"
    return (
        f"The user rated these movies, most recent first, from 0.5 to 5 stars:\n{rated}\n\n"
        f"{task}:\n{listed}\n\n"
        f'Answer with {{"ranking": [...]}}, {answer}.'
    )


def messages(
    history: Sequence[tuple[str, float]],
    candidates: Sequence[str],
    examples: Sequence[tuple[str, list]] = (),
    style: str = "numbers",
    top_n: int | None = None,
) -> list[dict[str, str]]:
    """Chat messages for one ranking: the system prompt, any worked examples, the request.

    Each example is a user message built with `user_message` and the list of labels that
    answers it.
    """
    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    for prompt, ranking in examples:
        chat.append({"role": "user", "content": prompt})
        chat.append({"role": "assistant", "content": json.dumps({"ranking": ranking})})
    chat.append({"role": "user", "content": user_message(history, candidates, style, top_n)})
    return chat


def answer_labels(positions: Sequence[int], style: str = "numbers") -> list:
    """The labels for positions counted from 1, as the model should write them."""
    if style == "numbers":
        return list(positions)
    return [ascii_uppercase[position - 1] for position in positions]


def parse_ranking(
    answer: str, n: int, style: str = "numbers", top_n: int | None = None
) -> tuple[list[int], int, int]:
    """Read positions 1..n, best first, from a model answer.

    Returns the ranking, the number of invalid or repeated entries that were dropped, and
    the number of candidates that had to be filled in. The ranking always holds `top_n`
    distinct positions, or all n without `top_n`: missing ones are filled in the order the
    candidates were shown, and entries beyond `top_n` are ignored.
    """
    labels = candidate_labels(n, style)
    position_of = {label: index for index, label in enumerate(labels, 1)}
    try:
        values = json.loads(answer)["ranking"]
        if not isinstance(values, list):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        pattern = r"-?\d+" if style == "numbers" else r"\b[A-Za-z]\b"
        values = re.findall(pattern, answer)
    wanted = n if top_n is None else top_n
    ranking, dropped = [], 0
    for value in values:
        position = position_of.get(str(value).strip().upper())
        if position is None or position in ranking:
            dropped += 1
        elif len(ranking) < wanted:
            ranking.append(position)
    filled = [position for position in range(1, n + 1) if position not in ranking]
    missing = wanted - len(ranking)
    return ranking + filled[:missing], dropped, missing


def borda(rankings: Sequence[Sequence[int]]) -> list[int]:
    """Combine rankings by mean position, ties by movie ID.

    The rankings may be top lists of different movies, all of the same length. A movie
    absent from a ranking counts as placed just after its end.
    """
    if not rankings:
        return []
    length = len(rankings[0])
    if any(len(ranking) != length or len(set(ranking)) != length for ranking in rankings):
        raise ValueError("Every ranking must have the same length and no repeated movie.")
    movies = set().union(*rankings)
    position = {movie: 0 for movie in movies}
    for ranking in rankings:
        index = {movie: place for place, movie in enumerate(ranking)}
        for movie in movies:
            position[movie] += index.get(movie, length)
    return sorted(movies, key=lambda movie: (position[movie], movie))


def mean_kendall_tau(rankings: Sequence[Sequence[int]]) -> float:
    """Average Kendall's τ over every pair of complete rankings of the same movies.

    1 means the order of the candidates never changed the answer.
    """
    if len(rankings) < 2:
        raise ValueError("Kendall's τ needs at least two rankings.")
    movies = sorted(rankings[0])
    taus = []
    for first, second in combinations(rankings, 2):
        first_position = {movie: index for index, movie in enumerate(first)}
        second_position = {movie: index for index, movie in enumerate(second)}
        tau = kendalltau(
            [first_position[movie] for movie in movies],
            [second_position[movie] for movie in movies],
        ).statistic
        taus.append(tau)
    return float(np.mean(taus))


def top_overlap(rankings: Sequence[Sequence[int]], k: int = 10) -> float:
    """Average share of the top k that two rankings have in common, over every pair.

    1 means the same k movies on top whatever the order of the candidates. Works for
    complete rankings and for top lists alike.
    """
    if len(rankings) < 2:
        raise ValueError("The overlap needs at least two rankings.")
    shared = [len(set(a[:k]) & set(b[:k])) / k for a, b in combinations(rankings, 2)]
    return float(np.mean(shared))
