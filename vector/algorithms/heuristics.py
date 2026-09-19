"""Heuristic functions for informed search (A*, Greedy Best-First).

Kept separate from common.py since heuristics are a distinct concept
(estimating remaining cost) from path reconstruction mechanics.
"""

from __future__ import annotations

import math
from typing import Callable

from vector.grid import Position

Heuristic = Callable[[Position, Position], float]


def manhattan_distance(a: Position, b: Position) -> float:
    """L1 distance. Admissible on a 4-connected grid with min cost 1."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a: Position, b: Position) -> float:
    """L2 distance. Also admissible, but a looser (less informed) bound
    than Manhattan distance on a 4-connected grid, since real travel
    distance can never be less than the Manhattan distance there."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


HEURISTICS: dict[str, Heuristic] = {
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
}


def get_heuristic(name: str) -> Heuristic:
    """Look up a heuristic function by name.

    Raises:
        ValueError: if name is not a registered heuristic.
    """
    try:
        return HEURISTICS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(HEURISTICS))
        raise ValueError(f"Unknown heuristic '{name}'. Valid options: {valid}") from exc
