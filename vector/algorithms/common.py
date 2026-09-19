"""Helpers shared across algorithm implementations.

Kept separate from algorithms/__init__.py (which defines the
SearchResult contract) so each algorithm module only imports what it
needs: the contract, and this reconstruction helper.
"""

from __future__ import annotations

from vector.grid import Position


def reconstruct_path(
    parents: dict[Position, Position], start: Position, goal: Position
) -> list[Position]:
    """Rebuild the path from start to goal using recorded parent links.

    Args:
        parents: Maps each visited position to the position it was
            reached from. Must not contain an entry for `start`.
        start: The search's start position.
        goal: The search's goal position. Must be a key in `parents`,
            or equal to `start`.

    Returns:
        Ordered list of positions from start to goal, inclusive.
    """
    if goal == start:
        return [start]

    path = [goal]
    current = goal
    while current != start:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path
