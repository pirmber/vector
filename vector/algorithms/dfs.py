"""Depth-First Search.

DFS explores as deep as possible along each branch before backtracking,
using a LIFO stack. It gives no shortest-path guarantee of any kind —
included primarily as a contrast case to show what happens without a
frontier ordering strategy.
"""

from __future__ import annotations

import time

from vector.algorithms import SearchResult
from vector.algorithms.common import reconstruct_path
from vector.grid import Grid, Position


def search(grid: Grid, start: Position, goal: Position) -> SearchResult:
    """Run iterative DFS from start to goal on grid."""
    start_time = time.perf_counter()

    if start == goal:
        elapsed = time.perf_counter() - start_time
        return SearchResult(
            algorithm_name="DFS",
            success=True,
            path=[start],
            nodes_explored=1,
            max_frontier_size=1,
            execution_time_seconds=elapsed,
        )

    stack: list[Position] = [start]
    visited: set[Position] = {start}
    parents: dict[Position, Position] = {}

    nodes_explored = 0
    max_frontier_size = 1
    explored_order: list[Position] = []

    while stack:
        max_frontier_size = max(max_frontier_size, len(stack))
        current = stack.pop()
        nodes_explored += 1
        explored_order.append(current)

        if current == goal:
            elapsed = time.perf_counter() - start_time
            path = reconstruct_path(parents, start, goal)
            total_cost = sum(grid.cost_at(pos) for pos in path[1:])
            return SearchResult(
                algorithm_name="DFS",
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
                stack.append(neighbor)

    elapsed = time.perf_counter() - start_time
    return SearchResult(
        algorithm_name="DFS",
        success=False,
        nodes_explored=nodes_explored,
        max_frontier_size=max_frontier_size,
        execution_time_seconds=elapsed,
        explored_order=explored_order,
    )
