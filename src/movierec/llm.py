"""Prompts, answer parsing and rank aggregation for the LLM ranker.

Nothing here calls a model, so every piece can be tested without a GPU.
"""

import json
import re
from collections.abc import Sequence
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau

SYSTEM_PROMPT = (
    "You are a movie recommender. You rank candidate movies by how likely a user is to "
    "enjoy them, judging only from the movies they rated. Answer with JSON only."
)

# Ollama accepts a JSON schema in `format`, which keeps the answer parseable.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
}


def describe(title: str, genres: Sequence[str] | None) -> str:
    """One line per movie: the MovieLens title, which ends with the year, and its genres."""
    return f"{title} [{', '.join(genres)}]" if genres else title


def user_message(history: Sequence[tuple[str, float]], candidates: Sequence[str]) -> str:
    """Ask to rank `candidates`, shown with numbers from 1, given the rated `history`.

    `history` holds (description, rating) pairs, most recent first.
    """
    rated = "\n".join(f"- {movie}: {rating:g}" for movie, rating in history)
    listed = "\n".join(f"{number}. {movie}" for number, movie in enumerate(candidates, 1))
    n = len(candidates)
    return (
        f"The user rated these movies, most recent first, from 0.5 to 5 stars:\n{rated}\n\n"
        f"Rank these {n} candidate movies from most to least likely to be enjoyed by the user:\n"
        f"{listed}\n\n"
        f'Answer with {{"ranking": [...]}}, listing every number from 1 to {n} exactly once, '
        "best first."
    )


def messages(
    history: Sequence[tuple[str, float]],
    candidates: Sequence[str],
    examples: Sequence[tuple[str, list[int]]] = (),
) -> list[dict[str, str]]:
    """Chat messages for one ranking: the system prompt, any worked examples, the request.

    Each example is a user message built with `user_message` and the ranking that answers it.
    """
    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    for prompt, ranking in examples:
        chat.append({"role": "user", "content": prompt})
        chat.append({"role": "assistant", "content": json.dumps({"ranking": ranking})})
    chat.append({"role": "user", "content": user_message(history, candidates)})
    return chat


def parse_ranking(answer: str, n: int) -> tuple[list[int], int, int]:
    """Read a ranking of the numbers 1..n from a model answer.

    Returns the complete ranking, the number of invalid or repeated entries that were
    dropped, and the number of missing candidates. Missing candidates are appended in the
    order they were shown, so the ranking always holds each number exactly once.
    """
    try:
        values = json.loads(answer)["ranking"]
        if not isinstance(values, list):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        values = re.findall(r"-?\d+", answer)
    ranking, dropped = [], 0
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if 1 <= number <= n and number not in ranking:
            ranking.append(number)
        else:
            dropped += 1
    missing = [number for number in range(1, n + 1) if number not in ranking]
    return ranking + missing, dropped, len(missing)


def borda(rankings: Sequence[Sequence[int]]) -> list[int]:
    """Combine rankings of the same movies by their mean position, ties by movie ID."""
    if not rankings:
        return []
    movies = set(rankings[0])
    if any(set(ranking) != movies or len(ranking) != len(movies) for ranking in rankings):
        raise ValueError("Every ranking must hold the same movies exactly once.")
    position = {movie: 0 for movie in movies}
    for ranking in rankings:
        for index, movie in enumerate(ranking):
            position[movie] += index
    return sorted(movies, key=lambda movie: (position[movie], movie))


def mean_kendall_tau(rankings: Sequence[Sequence[int]]) -> float:
    """Average Kendall's τ over every pair of rankings of the same movies.

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
