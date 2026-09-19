"""Generate analysis graphs from real benchmark runs (never hardcoded).

Run with:
    PYTHONPATH=. python experiments/analyze.py
Produces PNGs into experiments/output/ and a raw CSV of every run.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from vector.benchmark import ALGORITHMS, run_all, write_csv
from vector.environment import generate_random_environment

OUTPUT_DIR = Path(__file__).parent / "output"
CSV_PATH = OUTPUT_DIR / "benchmark_results.csv"


def sweep_grid_size(sizes: list[int], density: float, seeds: list[int]) -> dict[str, dict[int, list]]:
    """Run every algorithm across grid sizes and seeds. Returns
    {algorithm_name: {size: [SearchResult, ...]}} for successful runs only.
    """
    results_by_algo: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for size in sizes:
        for seed in seeds:
            grid = generate_random_environment(width=size, height=size, obstacle_density=density, seed=seed)
            results = run_all(grid)
            write_csv(results, str(CSV_PATH), extra_fields={"size": size, "density": density, "seed": seed})
            for r in results:
                if r.success:
                    results_by_algo[r.algorithm_name][size].append(r)

    return results_by_algo


def sweep_density(sizes_fixed: int, densities: list[float], seeds: list[int]) -> dict[str, dict[float, list]]:
    results_by_algo: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))

    for density in densities:
        for seed in seeds:
            grid = generate_random_environment(width=sizes_fixed, height=sizes_fixed, obstacle_density=density, seed=seed)
            results = run_all(grid)
            write_csv(results, str(CSV_PATH), extra_fields={"size": sizes_fixed, "density": density, "seed": seed})
            for r in results:
                results_by_algo[r.algorithm_name][density].append(r)

    return results_by_algo


def plot_size_vs_metric(results_by_algo: dict, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo_name, by_size in results_by_algo.items():
        sizes = sorted(by_size.keys())
        means = [statistics.mean(getattr(r, metric) for r in by_size[s]) for s in sizes if by_size[s]]
        valid_sizes = [s for s in sizes if by_size[s]]
        if valid_sizes:
            ax.plot(valid_sizes, means, marker="o", label=algo_name)
    ax.set_xlabel("Grid size (N x N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_density_vs_success_rate(results_by_algo: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo_name, by_density in results_by_algo.items():
        densities = sorted(by_density.keys())
        rates = []
        for d in densities:
            runs = by_density[d]
            rates.append(sum(1 for r in runs if r.success) / len(runs) if runs else 0)
        ax.plot(densities, rates, marker="o", label=algo_name)
    ax.set_xlabel("Obstacle density")
    ax.set_ylabel("Success rate")
    ax.set_title("Obstacle density vs success rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_algo_bar(results_by_algo: dict, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    names = list(results_by_algo.keys())
    means = []
    for name in names:
        all_runs = [r for size_runs in results_by_algo[name].values() for r in size_runs]
        means.append(statistics.mean(getattr(r, metric) for r in all_runs) if all_runs else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, means, color="#3a6ea5")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        CSV_PATH.unlink()

    sizes = [10, 20, 30, 40, 50]
    seeds = [1, 2, 3, 4, 5]
    density = 0.25

    print("Running grid-size sweep...")
    size_results = sweep_grid_size(sizes, density, seeds)
    plot_size_vs_metric(
        size_results, "execution_time_seconds", "Mean execution time (s)",
        "Grid size vs execution time", OUTPUT_DIR / "size_vs_time.png",
    )
    plot_size_vs_metric(
        size_results, "nodes_explored", "Mean nodes explored",
        "Grid size vs nodes explored", OUTPUT_DIR / "size_vs_explored.png",
    )
    plot_algo_bar(
        size_results, "total_cost", "Mean path cost",
        "Algorithm vs path cost (successful runs)", OUTPUT_DIR / "algo_vs_cost.png",
    )
    plot_algo_bar(
        size_results, "nodes_explored", "Mean nodes explored",
        "Algorithm vs nodes explored (successful runs)", OUTPUT_DIR / "algo_vs_explored.png",
    )

    print("Running density sweep...")
    densities = [0.05, 0.15, 0.25, 0.35, 0.45]
    density_results = sweep_density(40, densities, seeds)
    plot_density_vs_success_rate(density_results, OUTPUT_DIR / "density_vs_success.png")

    print(f"Graphs written to {OUTPUT_DIR}/")
    print(f"Raw data written to {CSV_PATH}")


if __name__ == "__main__":
    main()
