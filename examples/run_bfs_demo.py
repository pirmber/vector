"""Minimal end-to-end demo: generate an environment, run BFS, print results.

Run with:
    python examples/run_bfs_demo.py
"""

from __future__ import annotations

from vector.algorithms import bfs
from vector.environment import generate_random_environment


def main() -> None:
    grid = generate_random_environment(
        width=10, height=6, obstacle_density=0.25, seed=42
    )

    print("Environment (seed=42, 10x6, density=0.25):")
    print(grid.to_ascii())
    print()

    result = bfs.search(grid, grid.start, grid.goal)

    print(f"Algorithm:        {result.algorithm_name}")
    print(f"Success:          {result.success}")
    print(f"Path length:      {result.path_length} steps")
    print(f"Total cost:       {result.total_cost}")
    print(f"Nodes explored:   {result.nodes_explored}")
    print(f"Max frontier:     {result.max_frontier_size}")
    print(f"Execution time:   {result.execution_time_seconds * 1000:.3f} ms")
    if result.success:
        print(f"Path:             {result.path}")


if __name__ == "__main__":
    main()
