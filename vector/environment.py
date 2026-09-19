"""Environment generation for VECTOR.

Generation logic is kept separate from the Grid data structure itself
so that new generation strategies (weighted terrain, dynamic obstacles,
hand-authored maps) can be added without changing how algorithms
consume a Grid.
"""

from __future__ import annotations

import random

from vector.grid import CellType, Grid, Position

# Default terrain cost tiers, keyed by the same symbols used in ASCII maps.
DEFAULT_TERRAIN_COSTS: dict[str, float] = {".": 1.0, "~": 3.0, "^": 6.0}


def generate_random_environment(
    width: int,
    height: int,
    obstacle_density: float = 0.3,
    seed: int | None = None,
    start: Position | None = None,
    goal: Position | None = None,
    terrain_weights: dict[str, float] | None = None,
) -> Grid:
    """Generate a random grid environment with reproducible output.

    Args:
        width: Number of columns.
        height: Number of rows.
        obstacle_density: Fraction of non-start/goal cells that should be
            blocked, in [0.0, 1.0).
        seed: Random seed. The same seed with the same parameters always
            produces the same environment.
        start: Start position. Defaults to top-left corner (0, 0).
        goal: Goal position. Defaults to bottom-right corner.
        terrain_weights: Optional probability weights for non-blocked
            terrain symbols among {".", "~", "^"}, e.g. {".": 0.7,
            "~": 0.2, "^": 0.1}. Weights need not sum to 1 (they are
            normalized). If omitted, every traversable cell costs 1.0
            (i.e. all weight on ".").

    Returns:
        A Grid with obstacles and (optionally) weighted terrain placed,
        with start/goal cells set.

    Raises:
        ValueError: for invalid dimensions/density/weights, or if
            start/goal coincide or fall outside the grid.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not (0.0 <= obstacle_density < 1.0):
        raise ValueError("obstacle_density must be in [0.0, 1.0)")

    start = start if start is not None else (0, 0)
    goal = goal if goal is not None else (height - 1, width - 1)

    grid = Grid(width=width, height=height)

    if not grid.in_bounds(start):
        raise ValueError(f"start {start} is outside the grid")
    if not grid.in_bounds(goal):
        raise ValueError(f"goal {goal} is outside the grid")
    if start == goal:
        raise ValueError("start and goal must be different positions")

    terrain_weights = terrain_weights or {".": 1.0}
    unknown = set(terrain_weights) - set(DEFAULT_TERRAIN_COSTS)
    if unknown:
        raise ValueError(f"unknown terrain symbols: {sorted(unknown)}")
    total_weight = sum(terrain_weights.values())
    if total_weight <= 0:
        raise ValueError("terrain_weights must sum to a positive value")

    symbols = list(terrain_weights.keys())
    probabilities = [w / total_weight for w in terrain_weights.values()]

    rng = random.Random(seed)

    for row in range(height):
        for col in range(width):
            pos = (row, col)
            if pos == start or pos == goal:
                continue
            if rng.random() < obstacle_density:
                grid.set_cell(pos, CellType.BLOCKED)
            else:
                symbol = rng.choices(symbols, weights=probabilities, k=1)[0]
                grid.set_cell(pos, CellType.TRAVERSABLE, cost=DEFAULT_TERRAIN_COSTS[symbol])

    grid.set_cell(start, CellType.START)
    grid.set_cell(goal, CellType.GOAL)

    return grid


def from_ascii(lines: list[str]) -> Grid:
    """Build a Grid from an explicit ASCII map.

    Supports: '.' traversable (cost 1), '~' traversable (cost 3),
    '^' traversable (cost 6), '#' blocked, 'S' start, 'E' goal. Useful
    for tests and for hand-authored example maps.
    """
    if not lines:
        raise ValueError("lines must be non-empty")

    height = len(lines)
    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError("all rows must have the same width")

    symbol_to_type = {
        ".": CellType.TRAVERSABLE,
        "~": CellType.TRAVERSABLE,
        "^": CellType.TRAVERSABLE,
        "#": CellType.BLOCKED,
        "S": CellType.START,
        "E": CellType.GOAL,
    }

    grid = Grid(width=width, height=height)
    for row, line in enumerate(lines):
        for col, symbol in enumerate(line):
            if symbol not in symbol_to_type:
                raise ValueError(f"unknown symbol '{symbol}' at ({row}, {col})")
            cost = DEFAULT_TERRAIN_COSTS.get(symbol, 1.0)
            grid.set_cell((row, col), symbol_to_type[symbol], cost=cost)

    if grid.start is None:
        raise ValueError("map has no start ('S') cell")
    if grid.goal is None:
        raise ValueError("map has no goal ('E') cell")

    return grid
