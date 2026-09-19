"""Benchmarking engine: run every algorithm on the same environment and
collect comparable metrics. Deliberately does not judge or rank
algorithms — it only measures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from vector.algorithms import SearchResult, astar, bfs, dfs, dijkstra, greedy
from vector.environment import generate_random_environment
from vector.grid import Grid, Position

# Registry of runnable algorithms. Each entry is (label, callable).
# A* and Greedy are parameterized by heuristic, so they're expanded
# into separate labeled entries rather than special-cased.
ALGORITHMS: dict[str, callable] = {
    "BFS": lambda g, s, t: bfs.search(g, s, t),
    "DFS": lambda g, s, t: dfs.search(g, s, t),
    "Dijkstra": lambda g, s, t: dijkstra.search(g, s, t),
    "A* (manhattan)": lambda g, s, t: astar.search(g, s, t, heuristic="manhattan"),
    "A* (euclidean)": lambda g, s, t: astar.search(g, s, t, heuristic="euclidean"),
    "Greedy (manhattan)": lambda g, s, t: greedy.search(g, s, t, heuristic="manhattan"),
}


def run_all(grid: Grid, start: Position | None = None, goal: Position | None = None) -> list[SearchResult]:
    """Run every registered algorithm on the same grid/start/goal.

    Returns:
        List of SearchResult, one per algorithm, in registry order.
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal
    if start is None or goal is None:
        raise ValueError("grid has no start/goal and none were provided")

    return [fn(grid, start, goal) for fn in ALGORITHMS.values()]


def format_table(results: list[SearchResult]) -> str:
    """Render results as an aligned text table."""
    header = f"{'Algorithm':<20}{'Success':<10}{'Cost':<10}{'Steps':<8}{'Explored':<10}{'MaxFront':<10}{'Time(ms)':<10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.algorithm_name:<20}{str(r.success):<10}{r.total_cost:<10.2f}"
            f"{r.path_length:<8}{r.nodes_explored:<10}{r.max_frontier_size:<10}"
            f"{r.execution_time_seconds * 1000:<10.3f}"
        )
    return "\n".join(lines)


def write_csv(results: list[SearchResult], path: str, extra_fields: dict | None = None) -> None:
    """Append one row per result to a CSV file, creating it with a header if needed.

    Args:
        results: Results to write.
        path: Destination CSV path.
        extra_fields: Extra columns (e.g. grid size, density, seed) applied
            identically to every row, useful for later cross-run analysis.
    """
    extra_fields = extra_fields or {}
    file_path = Path(path)
    file_exists = file_path.exists()

    fieldnames = list(extra_fields.keys()) + [
        "algorithm_name",
        "success",
        "total_cost",
        "path_length",
        "nodes_explored",
        "max_frontier_size",
        "execution_time_seconds",
    ]

    with file_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            row = dict(extra_fields)
            row.update(
                {
                    "algorithm_name": r.algorithm_name,
                    "success": r.success,
                    "total_cost": r.total_cost,
                    "path_length": r.path_length,
                    "nodes_explored": r.nodes_explored,
                    "max_frontier_size": r.max_frontier_size,
                    "execution_time_seconds": r.execution_time_seconds,
                }
            )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="VECTOR benchmark runner")
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default=None, help="optional path to append results to")
    args = parser.parse_args()

    grid = generate_random_environment(
        width=args.width, height=args.height, obstacle_density=args.density, seed=args.seed
    )
    results = run_all(grid)

    print("VECTOR BENCHMARK")
    print("-" * 60)
    print(f"Environment: {args.width} x {args.height}")
    print(f"Obstacle density: {args.density:.0%}")
    print(f"Seed: {args.seed}")
    print()
    print(format_table(results))

    if args.csv:
        write_csv(
            results,
            args.csv,
            extra_fields={
                "width": args.width,
                "height": args.height,
                "density": args.density,
                "seed": args.seed,
            },
        )
        print(f"\nResults appended to {args.csv}")


if __name__ == "__main__":
    main()
