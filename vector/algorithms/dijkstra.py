"""Dijkstra's algorithm.

Uses a min-priority queue keyed on accumulated cost g(n). Guarantees
the minimum-cost path on any non-negative-weight grid, unlike BFS/DFS
which are cost-blind. This is the first algorithm here that actually
uses `grid.cost_at`.
"""

from __future__ import annotations

import heapq
import itertools
import time

from vector.algorithms import SearchResult
from vector.algorithms.common import reconstruct_path
from vector.grid import Grid, Position


def search(grid: Grid, start: Position, goal: Position) -> SearchResult:
    """Run Dijkstra's algorithm from start to goal on grid."""
    start_time = time.perf_counter()

    if start == goal:
        elapsed = time.perf_counter() - start_time
        return SearchResult(
            algorithm_name="Dijkstra",
            success=True,
            path=[start],
            nodes_explored=1,
            max_frontier_size=1,
            execution_time_seconds=elapsed,
        )

    counter = itertools.count()  # stable tie-breaker, avoids comparing Positions
    frontier: list[tuple[float, int, Position]] = [(0.0, next(counter), start)]
    best_cost: dict[Position, float] = {start: 0.0}
    parents: dict[Position, Position] = {}
    visited: set[Position] = set()

    nodes_explored = 0
    max_frontier_size = 1
    explored_order: list[Position] = []

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        cost, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1
        explored_order.append(current)

        if current == goal:
            elapsed = time.perf_counter() - start_time
            path = reconstruct_path(parents, start, goal)
            return SearchResult(
                algorithm_name="Dijkstra",
                success=True,
                path=path,
                total_cost=cost,
                nodes_explored=nodes_explored,
                max_frontier_size=max_frontier_size,
                execution_time_seconds=elapsed,
                explored_order=explored_order,
            )

        for neighbor in grid.neighbors(current):
            new_cost = cost + grid.cost_at(neighbor)
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parents[neighbor] = current
                heapq.heappush(frontier, (new_cost, next(counter), neighbor))

    elapsed = time.perf_counter() - start_time
    return SearchResult(
        algorithm_name="Dijkstra",
        success=False,
        nodes_explored=nodes_explored,
        max_frontier_size=max_frontier_size,
        execution_time_seconds=elapsed,
        explored_order=explored_order,
    )
