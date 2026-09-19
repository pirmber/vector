"""Breadth-First Search.

BFS explores the grid in expanding "rings" of distance from the start,
using a FIFO queue. Because it always expands the shallowest unexplored
node first, it is guaranteed to find a shortest path *in number of
steps* on an unweighted (uniform-cost) grid. It has no notion of
movement cost, so on weighted terrain it does not guarantee the
cheapest path — only the one with fewest edges.
"""

from __future__ import annotations

import time
from collections import deque

from vector.algorithms import SearchResult
from vector.algorithms.common import reconstruct_path
from vector.grid import Grid, Position


def search(grid: Grid, start: Position, goal: Position) -> SearchResult:
    """Run BFS from start to goal on grid.

    Args:
        grid: The environment to search.
        start: Start position.
        goal: Goal position.

    Returns:
        A SearchResult describing the outcome and search metrics.
    """
    start_time = time.perf_counter()

    if start == goal:
        elapsed = time.perf_counter() - start_time
        return SearchResult(
            algorithm_name="BFS",
            success=True,
            path=[start],
            total_cost=0.0,
            nodes_explored=1,
            max_frontier_size=1,
            execution_time_seconds=elapsed,
        )

    frontier: deque[Position] = deque([start])
    visited: set[Position] = {start}
    parents: dict[Position, Position] = {}

    nodes_explored = 0
    max_frontier_size = 1
    explored_order: list[Position] = []

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current = frontier.popleft()
        nodes_explored += 1
        explored_order.append(current)

        if current == goal:
            elapsed = time.perf_counter() - start_time
            path = reconstruct_path(parents, start, goal)
            total_cost = sum(grid.cost_at(pos) for pos in path[1:])
            return SearchResult(
                algorithm_name="BFS",
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
                frontier.append(neighbor)

    elapsed = time.perf_counter() - start_time
    return SearchResult(
        algorithm_name="BFS",
        success=False,
        nodes_explored=nodes_explored,
        max_frontier_size=max_frontier_size,
        execution_time_seconds=elapsed,
        explored_order=explored_order,
    )
