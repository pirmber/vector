"""Greedy Best-First Search.

f(n) = h(n) only — no accumulated cost term. This makes it fast to
reach a goal-looking region but gives NO optimality guarantee: it can
easily be lured down a costly or roundabout path by ignoring what it
has already spent, which is exactly the behavior this project uses it
to demonstrate against A*/Dijkstra.
"""

from __future__ import annotations

import heapq
import itertools
import time

from vector.algorithms import SearchResult
from vector.algorithms.common import reconstruct_path
from vector.algorithms.heuristics import get_heuristic
from vector.grid import Grid, Position


def search(
    grid: Grid, start: Position, goal: Position, heuristic: str = "manhattan"
) -> SearchResult:
    """Run Greedy Best-First Search from start to goal on grid."""
    h = get_heuristic(heuristic)
    start_time = time.perf_counter()

    if start == goal:
        elapsed = time.perf_counter() - start_time
        return SearchResult(
            algorithm_name=f"Greedy ({heuristic})",
            success=True,
            path=[start],
            nodes_explored=1,
            max_frontier_size=1,
            execution_time_seconds=elapsed,
        )

    counter = itertools.count()
    frontier: list[tuple[float, int, Position]] = [(h(start, goal), next(counter), start)]
    parents: dict[Position, Position] = {}
    visited: set[Position] = {start}

    nodes_explored = 0
    max_frontier_size = 1
    explored_order: list[Position] = []

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        _, _, current = heapq.heappop(frontier)
        nodes_explored += 1
        explored_order.append(current)

        if current == goal:
            elapsed = time.perf_counter() - start_time
            path = reconstruct_path(parents, start, goal)
            total_cost = sum(grid.cost_at(pos) for pos in path[1:])
            return SearchResult(
                algorithm_name=f"Greedy ({heuristic})",
                success=True,
                path=path,
                total_cost=total_cost,
                nodes_explored=nodes_explored,
                max_frontier_size=max_frontier_size,
                execution_time_seconds=elapsed,
                explored_order=explored_order,
            )

        for neighbor in grid.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = current
                heapq.heappush(frontier, (h(neighbor, goal), next(counter), neighbor))

    elapsed = time.perf_counter() - start_time
    return SearchResult(
        algorithm_name=f"Greedy ({heuristic})",
        success=False,
        nodes_explored=nodes_explored,
        max_frontier_size=max_frontier_size,
        execution_time_seconds=elapsed,
        explored_order=explored_order,
    )
