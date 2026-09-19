"""A* Search.

f(n) = g(n) + h(n), where g is accumulated cost from start and h is a
heuristic estimate of remaining cost to goal. With an admissible
heuristic (never overestimates), A* is guaranteed optimal like
Dijkstra, but typically explores fewer nodes because h(n) biases
expansion toward the goal.
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
    """Run A* from start to goal on grid.

    Args:
        heuristic: Name of a registered heuristic ("manhattan" or
            "euclidean"). See heuristics.py.
    """
    h = get_heuristic(heuristic)
    start_time = time.perf_counter()

    if start == goal:
        elapsed = time.perf_counter() - start_time
        return SearchResult(
            algorithm_name=f"A* ({heuristic})",
            success=True,
            path=[start],
            nodes_explored=1,
            max_frontier_size=1,
            execution_time_seconds=elapsed,
        )

    counter = itertools.count()
    frontier: list[tuple[float, int, Position]] = [(h(start, goal), next(counter), start)]
    g_cost: dict[Position, float] = {start: 0.0}
    parents: dict[Position, Position] = {}
    visited: set[Position] = set()

    nodes_explored = 0
    max_frontier_size = 1
    explored_order: list[Position] = []

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1
        explored_order.append(current)

        if current == goal:
            elapsed = time.perf_counter() - start_time
            path = reconstruct_path(parents, start, goal)
            return SearchResult(
                algorithm_name=f"A* ({heuristic})",
                success=True,
                path=path,
                total_cost=g_cost[current],
                nodes_explored=nodes_explored,
                max_frontier_size=max_frontier_size,
                execution_time_seconds=elapsed,
                explored_order=explored_order,
            )

        for neighbor in grid.neighbors(current):
            tentative_g = g_cost[current] + grid.cost_at(neighbor)
            if tentative_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = tentative_g
                parents[neighbor] = current
                f_score = tentative_g + h(neighbor, goal)
                heapq.heappush(frontier, (f_score, next(counter), neighbor))

    elapsed = time.perf_counter() - start_time
    return SearchResult(
        algorithm_name=f"A* ({heuristic})",
        success=False,
        nodes_explored=nodes_explored,
        max_frontier_size=max_frontier_size,
        execution_time_seconds=elapsed,
        explored_order=explored_order,
    )
