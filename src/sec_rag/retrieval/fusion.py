"""Merge ranked result lists from different searches."""

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Combine rankings by position, not by score.

    Dense and keyword scores are on unrelated scales, but ranks are comparable: each list
    adds 1 / (k + rank) for every item it contains, so items ranked well by several searches
    rise to the top. k=60 is the value from the original RRF paper; larger k flattens the
    advantage of the very top ranks. Ties keep the order of the first list.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
