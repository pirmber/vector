"""Search algorithm implementations for VECTOR.

Every algorithm in this package exposes the same function signature:

    search(grid: Grid, start: Position, goal: Position, **kwargs) -> SearchResult

This uniform contract is what makes the benchmarking engine (Phase 5)
able to run any algorithm over any grid interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vector.grid import Position


@dataclass(frozen=True)
class SearchResult:
    """The outcome of running a search algorithm on a grid.

    Attributes:
        algorithm_name: Human-readable name, e.g. "BFS".
        success: True if a path from start to goal was found.
        path: Ordered list of positions from start to goal (inclusive).
            Empty if success is False.
        total_cost: Sum of movement costs along the path. For unweighted
            algorithms (BFS/DFS) this equals path length in steps when
            all costs are 1.0.
        nodes_explored: Number of nodes popped from the frontier and
            expanded (i.e. actually visited, not just enqueued).
        max_frontier_size: The largest size the frontier (queue/stack/
            heap) reached during the run.
        execution_time_seconds: Wall-clock time for the search itself,
            excluding grid generation.
        explored_order: Positions in the order they were popped from
            the frontier and expanded. Used for animation; not required
            for benchmarking, which only needs the count.
    """

    algorithm_name: str
    success: bool
    path: list[Position] = field(default_factory=list)
    total_cost: float = 0.0
    nodes_explored: int = 0
    max_frontier_size: int = 0
    execution_time_seconds: float = 0.0
    explored_order: list[Position] = field(default_factory=list)

    @property
    def path_length(self) -> int:
        """Number of steps in the path (edges, not nodes)."""
        return max(len(self.path) - 1, 0)
