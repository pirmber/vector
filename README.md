# VECTOR — Intelligent Pathfinding & Decision Engine

VECTOR generates grid environments and runs multiple search algorithms —
implemented from scratch, no pathfinding libraries — through the *same*
environment, then measures and compares their behavior. It is not a
single A* implementation; it's a small research harness for observing
the real trade-offs between search strategies.

## Motivation

Most pathfinding tutorials show one algorithm and declare it "the
answer." In practice, BFS, DFS, Dijkstra, A*, and Greedy Best-First
Search each make different trade-offs between optimality, speed, and
memory — and those trade-offs only become visible when you run them
side by side on identical inputs and measure what actually happens.
VECTOR exists to make those trade-offs empirical rather than
theoretical.

**This project does not declare a "best" algorithm.** Every claim below
about an algorithm's behavior is backed by a benchmark run included in
this repository (`experiments/output/`), not asserted from theory alone.

## Architecture

```
Environment Generator  (vector/environment.py)
        │  seeded random obstacles + weighted terrain
        ▼
Grid Representation     (vector/grid.py)
        │  cell types, movement costs, neighbor logic
        ▼
Pathfinding Algorithms   (vector/algorithms/*)
        │  BFS · DFS · Dijkstra · A* · Greedy — uniform interface
        ▼
Path Reconstruction     (vector/algorithms/common.py)
        │  parent-pointer walk from goal to start
        ▼
Benchmarking Engine      (vector/benchmark.py)
        │  same map, every algorithm, same metrics
        ▼
Visualization + Analysis (vector/visualization.py, experiments/analyze.py)
```

An optional outer layer, `vector/dynamic.py`, wraps this pipeline to
support obstacles changing mid-execution and triggering re-planning,
without modifying any of the static core above it.

## Algorithms

Every algorithm implements the same contract:

```python
def search(grid: Grid, start: Position, goal: Position, **kwargs) -> SearchResult
```

`SearchResult` carries: `path`, `success`, `total_cost`, `nodes_explored`,
`max_frontier_size`, `execution_time_seconds`, and `explored_order` (for
animation). This uniform contract is what lets the benchmark engine run
any algorithm interchangeably.

### BFS — Breadth-First Search
Expands nodes in strict order of distance (in steps) from the start,
using a FIFO queue. Guarantees a shortest path *in number of edges* on
any grid — but is blind to movement cost, so on weighted terrain its
"shortest" path is not necessarily the cheapest.

### DFS — Depth-First Search
Expands as deep as possible along one branch before backtracking, using
a LIFO stack. No optimality guarantee of any kind. Included as a
contrast case: it often explores very few nodes but returns much longer
paths, since it commits to a direction rather than expanding uniformly.

### Dijkstra's Algorithm
Expands nodes in order of accumulated cost `g(n)`, using a min-priority
queue. Guarantees the minimum-cost path on any grid with non-negative
weights — the first algorithm here that correctly handles weighted
terrain.

### A* Search
Expands nodes in order of `f(n) = g(n) + h(n)`, where `g(n)` is the
accumulated cost from the start and `h(n)` is a heuristic estimate of
the remaining cost to the goal. With an **admissible** heuristic (one
that never overestimates true remaining cost), A* is guaranteed as
optimal as Dijkstra, but typically explores fewer nodes because `h(n)`
biases expansion toward the goal instead of expanding uniformly
outward. Supports two heuristics, selectable at call time:

- **Manhattan distance**: `|Δrow| + |Δcol|` — admissible and exact for
  4-connected movement with unit step cost.
- **Euclidean distance**: straight-line distance — also admissible, but
  a *looser* (less informed) lower bound on a 4-connected grid, since
  true travel distance there can never be less than Manhattan distance.
  A looser heuristic means less pruning, i.e. more nodes explored for
  the same optimal answer — demonstrated below.

### Greedy Best-First Search
Expands nodes in order of `f(n) = h(n)` only — no accumulated-cost term
at all. This makes it fast to move toward goal-looking regions, but
gives **no optimality guarantee**: it can be lured down an expensive or
roundabout route because it never accounts for what it has already
spent to get there.

### Why heuristic choice matters
`f(n) = g(n) + h(n)` only prunes search as effectively as `h(n)` is
*informative* — i.e., close to the true remaining cost without
exceeding it. A heuristic of zero everywhere degrades A* into Dijkstra.
A heuristic that overestimates can break optimality entirely. Manhattan
vs. Euclidean on a 4-connected grid is the smallest possible
demonstration of this: both are admissible, but the tighter one
(Manhattan) consistently explores fewer nodes for the identical optimal
answer (see results below).

## Benchmark methodology

For a given environment (fixed width, height, obstacle density, seed,
and optional terrain weights), every algorithm is run once against the
*exact same* grid, start, and goal. Measured per run:

- success (path found or not)
- total path cost (sum of terrain costs along the path)
- path length (steps/edges)
- nodes explored (nodes actually popped from the frontier, not merely
  enqueued)
- maximum frontier size reached
- wall-clock execution time (search only, excluding grid generation)

Results are written to CSV (`experiments/output/benchmark_results.csv`)
with one row per algorithm per run, tagged with the run's parameters
(size, density, seed), so cross-run analysis never depends on rerunning
anything.

## Experimental results

All numbers below come from actual runs in this repository — see
`experiments/analyze.py` for the exact sweep that produced them, and
`experiments/output/` for the generated CSV and PNGs.

**Uniform-cost grid (50×50, density 0.25, seed 1):**

| Algorithm | Success | Cost | Steps | Explored | Max Frontier | Time (ms) |
|---|---|---|---|---|---|---|
| BFS | True | 98.00 | 98 | 1873 | 49 | 3.09 |
| DFS | True | 160.00 | 160 | 194 | 90 | 0.21 |
| Dijkstra | True | 98.00 | 98 | 1873 | 49 | 4.53 |
| A* (manhattan) | True | 98.00 | 98 | **1129** | 176 | 3.35 |
| A* (euclidean) | True | 98.00 | 98 | 1698 | 78 | 4.85 |
| Greedy (manhattan) | True | 110.00 | 110 | **133** | 98 | 0.26 |

Observations directly supported by this run:
- BFS and Dijkstra find identical-cost paths and explore identical node
  counts here — expected, since with uniform cost, "fewest steps" and
  "cheapest" coincide, so Dijkstra's cost-ordering degenerates to BFS's
  distance-ordering.
- A* (manhattan) finds the same optimal cost as BFS/Dijkstra while
  exploring **40% fewer nodes** (1129 vs 1873).
- A* (euclidean) is still optimal, but explores more than A* (manhattan)
  — direct evidence that heuristic *tightness*, not just heuristic
  presence, determines pruning power.
- DFS and Greedy explore the fewest nodes by far, but both return
  measurably longer/costlier paths — the classic speed/optimality
  trade-off, here with actual numbers instead of a hand-wave.

**Weighted terrain (25×25, density 0.15, seed 3, terrain mix
70% cheap / 30% costly):**

| Algorithm | Cost | Explored |
|---|---|---|
| BFS | 86.0 | fewest steps, but ignores terrain cost |
| Dijkstra | 50.0 | accounts for terrain cost correctly |
| A* (manhattan) | 50.0 | same optimal cost, fewer nodes than Dijkstra |

Here BFS's "shortest" path is actively worse than Dijkstra/A*'s once
terrain cost is introduced — the clearest demonstration in this project
of *why* cost-aware search exists at all.

**Grid size vs. nodes explored** (`experiments/output/size_vs_explored.png`,
mean over 5 seeds per size): Dijkstra and BFS scale together and
steepest; A* (manhattan) scales visibly slower than both; Greedy stays
nearly flat across all tested sizes, at the cost of the suboptimal
paths shown above.

## Limitations

- Movement is 4-connected only (no diagonals); Euclidean heuristic is
  therefore looser than it would be on 8-connected movement.
- Environments are generated by independent per-cell obstacle
  probability, not guaranteed-solvable maze generation — some seeds
  produce disconnected start/goal pairs, which is treated as a valid
  experimental outcome (`success=False`), not an error.
- Dynamic re-planning (`vector/dynamic.py`) is a simple "detect
  invalidated path, re-plan from scratch" strategy, not an incremental
  re-planning algorithm (e.g. D* Lite) — intentionally, to keep the
  optional layer simple and decoupled from the static core.
- Benchmarks measure single-run wall-clock time on whatever machine
  runs them; no statistical significance testing is performed beyond
  averaging over a handful of seeds.

## Future improvements

- 8-connected movement and diagonal-aware heuristics (Chebyshev/octile).
- Bidirectional search and Jump Point Search as additional algorithms.
- Incremental re-planning (D* Lite) for the dynamic layer.
- Statistical rigor: confidence intervals over many more seeds per
  configuration.
- Weighted A* (`f(n) = g(n) + w·h(n)`) to explicitly explore the
  speed/optimality trade-off as a tunable parameter.

## Installation

```bash
git clone <this-repo>
cd vector
pip install -r requirements.txt
```

## Usage

Run a single benchmark:
```bash
PYTHONPATH=. python -m vector.benchmark --width 100 --height 100 --density 0.30 --seed 42 --csv results.csv
```

Run the minimal BFS demo:
```bash
PYTHONPATH=. python examples/run_bfs_demo.py
```

Run the dynamic re-planning demo:
```bash
PYTHONPATH=. python examples/run_dynamic_demo.py
```

Generate all analysis graphs from fresh benchmark data:
```bash
PYTHONPATH=. python experiments/analyze.py
```

Run the test suite:
```bash
PYTHONPATH=. python -m pytest tests/ -v
```

Visualize a single search:
```python
from vector.environment import generate_random_environment
from vector.algorithms import astar
from vector.visualization import save_result_plot

grid = generate_random_environment(width=30, height=20, obstacle_density=0.25, seed=3)
result = astar.search(grid, grid.start, grid.goal)
save_result_plot(grid, result, "output.png", explored_order=result.explored_order)
```

## Example output

```
VECTOR BENCHMARK
------------------------------------------------------------
Environment: 50 x 50
Obstacle density: 25%
Seed: 1

Algorithm           Success   Cost      Steps   Explored  MaxFront  Time(ms)
------------------------------------------------------------------------------
BFS                 True      98.00     98      1873      49        3.093
DFS                 True      160.00    160     194       90        0.211
Dijkstra            True      98.00     98      1873      49        4.534
A* (manhattan)      True      98.00     98      1129      176       3.353
A* (euclidean)      True      98.00     98      1698      78        4.854
Greedy (manhattan)  True      110.00    110     133       98        0.255
```

## Project structure

```
vector/
├── vector/
│   ├── grid.py              # Cell types, terrain costs, neighbor logic
│   ├── environment.py       # Seeded random generation + ASCII maps
│   ├── algorithms/
│   │   ├── bfs.py
│   │   ├── dfs.py
│   │   ├── dijkstra.py
│   │   ├── astar.py
│   │   ├── greedy.py
│   │   ├── heuristics.py    # Manhattan / Euclidean, pluggable
│   │   └── common.py        # Shared path reconstruction
│   ├── benchmark.py         # Runs every algorithm on identical maps
│   ├── visualization.py     # Static plots + animated GIFs (matplotlib)
│   └── dynamic.py           # Optional: obstacle changes + re-planning
├── experiments/
│   └── analyze.py           # Sweeps + graphs from real benchmark data
├── examples/
│   ├── run_bfs_demo.py
│   └── run_dynamic_demo.py
├── tests/
│   ├── test_core.py
│   └── test_dynamic.py
└── requirements.txt
```
