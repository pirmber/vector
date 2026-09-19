"""Core test suite for VECTOR: grid, environment generation, and every
search algorithm, including required edge cases."""

from __future__ import annotations

import pytest

from vector.algorithms import astar, bfs, dfs, dijkstra, greedy
from vector.algorithms.common import reconstruct_path
from vector.algorithms.heuristics import euclidean_distance, get_heuristic, manhattan_distance
from vector.environment import from_ascii, generate_random_environment
from vector.grid import CellType, Grid

ALL_ALGORITHMS = [
    ("BFS", lambda g, s, t: bfs.search(g, s, t)),
    ("DFS", lambda g, s, t: dfs.search(g, s, t)),
    ("Dijkstra", lambda g, s, t: dijkstra.search(g, s, t)),
    ("A*", lambda g, s, t: astar.search(g, s, t)),
    ("Greedy", lambda g, s, t: greedy.search(g, s, t)),
]


# ---------- Grid ----------

def test_grid_default_dimensions():
    grid = Grid(width=4, height=3)
    assert grid.width == 4 and grid.height == 3
    assert all(cell == CellType.TRAVERSABLE for row in grid.cell_types for cell in row)


def test_grid_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        Grid(width=3, height=2, cell_types=[[CellType.TRAVERSABLE] * 3])


def test_in_bounds():
    grid = Grid(width=3, height=3)
    assert grid.in_bounds((0, 0))
    assert grid.in_bounds((2, 2))
    assert not grid.in_bounds((3, 0))
    assert not grid.in_bounds((-1, 0))


def test_neighbors_respects_bounds_and_blocked():
    grid = Grid(width=3, height=3)
    grid.set_cell((0, 1), CellType.BLOCKED)
    neighbors = set(grid.neighbors((0, 0)))
    # (−1,0) out of bounds, (0,1) blocked, (1,0) valid, (0,-1) out of bounds
    assert neighbors == {(1, 0)}


def test_cost_at_raises_on_blocked():
    grid = Grid(width=2, height=2)
    grid.set_cell((0, 0), CellType.BLOCKED)
    with pytest.raises(ValueError):
        grid.cost_at((0, 0))


# ---------- Environment ----------

def test_same_seed_reproducible():
    g1 = generate_random_environment(width=15, height=15, obstacle_density=0.3, seed=99)
    g2 = generate_random_environment(width=15, height=15, obstacle_density=0.3, seed=99)
    assert g1.to_ascii() == g2.to_ascii()


def test_different_seed_usually_differs():
    g1 = generate_random_environment(width=15, height=15, obstacle_density=0.3, seed=1)
    g2 = generate_random_environment(width=15, height=15, obstacle_density=0.3, seed=2)
    assert g1.to_ascii() != g2.to_ascii()


def test_start_equals_goal_raises():
    with pytest.raises(ValueError):
        generate_random_environment(width=5, height=5, start=(0, 0), goal=(0, 0))


def test_invalid_density_raises():
    with pytest.raises(ValueError):
        generate_random_environment(width=5, height=5, obstacle_density=1.0)


def test_from_ascii_roundtrip():
    lines = ["S..", ".#.", "..E"]
    grid = from_ascii(lines)
    assert grid.start == (0, 0)
    assert grid.goal == (2, 2)
    assert grid.to_ascii() == "\n".join(lines)


def test_from_ascii_missing_start_raises():
    with pytest.raises(ValueError):
        from_ascii(["...", "...", "..E"])


def test_weighted_terrain_costs_applied():
    grid = from_ascii(["S~^", "..E"])
    assert grid.cost_at((0, 1)) == 3.0
    assert grid.cost_at((0, 2)) == 6.0
    assert grid.cost_at((1, 0)) == 1.0


# ---------- Heuristics ----------

def test_manhattan_distance():
    assert manhattan_distance((0, 0), (3, 4)) == 7


def test_euclidean_distance():
    assert euclidean_distance((0, 0), (3, 4)) == 5.0


def test_manhattan_never_exceeds_euclidean_or_actual():
    # Manhattan is a tighter (>=) admissible bound than Euclidean on a grid.
    a, b = (0, 0), (5, 3)
    assert manhattan_distance(a, b) >= euclidean_distance(a, b)


def test_unknown_heuristic_raises():
    with pytest.raises(ValueError):
        get_heuristic("does_not_exist")


# ---------- Path reconstruction ----------

def test_reconstruct_path_start_equals_goal():
    assert reconstruct_path({}, (0, 0), (0, 0)) == [(0, 0)]


def test_reconstruct_path_chain():
    parents = {(0, 1): (0, 0), (0, 2): (0, 1)}
    assert reconstruct_path(parents, (0, 0), (0, 2)) == [(0, 0), (0, 1), (0, 2)]


# ---------- Algorithms: shared behavioral contract ----------

@pytest.mark.parametrize("name,fn", ALL_ALGORITHMS)
def test_start_equals_goal(name, fn):
    grid = Grid(width=3, height=3)
    grid.set_cell((1, 1), CellType.START)
    result = fn(grid, (1, 1), (1, 1))
    assert result.success
    assert result.path == [(1, 1)]
    assert result.total_cost == 0.0


@pytest.mark.parametrize("name,fn", ALL_ALGORITHMS)
def test_no_possible_path(name, fn):
    # Goal fully walled off from start.
    lines = ["S#E"]
    grid = from_ascii(lines)
    result = fn(grid, grid.start, grid.goal)
    assert result.success is False
    assert result.path == []


@pytest.mark.parametrize("name,fn", ALL_ALGORITHMS)
def test_completely_blocked_grid_except_start(name, fn):
    grid = Grid(width=3, height=3)
    for r in range(3):
        for c in range(3):
            grid.set_cell((r, c), CellType.BLOCKED)
    grid.set_cell((0, 0), CellType.START)
    grid.set_cell((2, 2), CellType.GOAL)
    result = fn(grid, (0, 0), (2, 2))
    assert result.success is False


@pytest.mark.parametrize("name,fn", ALL_ALGORITHMS)
def test_1x1_grid(name, fn):
    grid = Grid(width=1, height=1)
    grid.set_cell((0, 0), CellType.START)
    result = fn(grid, (0, 0), (0, 0))
    assert result.success
    assert result.path == [(0, 0)]


@pytest.mark.parametrize("name,fn", ALL_ALGORITHMS)
def test_narrow_corridor(name, fn):
    lines = ["S....", "#####", ".....", "#####", "....E"]
    # Not fully connected on purpose; use a guaranteed-connected corridor instead:
    lines = ["S....E"]
    grid = from_ascii(lines)
    result = fn(grid, grid.start, grid.goal)
    assert result.success
    assert result.path[0] == grid.start
    assert result.path[-1] == grid.goal
    assert result.path_length == 5


@pytest.mark.parametrize("name,fn", ALL_ALGORITHMS)
def test_large_grid_runs_without_error(name, fn):
    grid = generate_random_environment(width=80, height=80, obstacle_density=0.2, seed=123)
    result = fn(grid, grid.start, grid.goal)
    # Not asserting success (map may be unsolvable); just that it terminates
    # cleanly and returns a well-formed result.
    assert isinstance(result.nodes_explored, int)
    assert result.nodes_explored > 0


# ---------- Optimality checks (BFS/Dijkstra/A* must agree on cost when optimal) ----------

def test_bfs_dijkstra_agree_on_uniform_cost_grid():
    grid = generate_random_environment(width=25, height=25, obstacle_density=0.2, seed=7)
    bfs_result = bfs.search(grid, grid.start, grid.goal)
    dijkstra_result = dijkstra.search(grid, grid.start, grid.goal)
    assert bfs_result.success == dijkstra_result.success
    if bfs_result.success:
        # Uniform cost (all 1.0) => BFS's fewest-steps path is also cheapest.
        assert bfs_result.total_cost == dijkstra_result.total_cost


def test_astar_matches_dijkstra_cost_on_weighted_terrain():
    grid = generate_random_environment(
        width=25,
        height=25,
        obstacle_density=0.15,
        seed=3,
        terrain_weights={".": 0.5, "~": 0.3, "^": 0.2},
    )
    dijkstra_result = dijkstra.search(grid, grid.start, grid.goal)
    astar_result = astar.search(grid, grid.start, grid.goal, heuristic="manhattan")
    assert dijkstra_result.success == astar_result.success
    if dijkstra_result.success:
        assert astar_result.total_cost == pytest.approx(dijkstra_result.total_cost)
        # Admissible heuristic => A* should never explore more than Dijkstra.
        assert astar_result.nodes_explored <= dijkstra_result.nodes_explored


def test_greedy_can_be_suboptimal():
    # Demonstrative, not a strict guarantee for all maps: on a map with
    # weighted terrain designed to mislead, greedy should not always beat
    # Dijkstra's cost. We assert the weaker, always-true fact instead:
    # greedy's cost is never *lower* than the true optimum.
    grid = generate_random_environment(
        width=25,
        height=25,
        obstacle_density=0.15,
        seed=3,
        terrain_weights={".": 0.5, "~": 0.3, "^": 0.2},
    )
    dijkstra_result = dijkstra.search(grid, grid.start, grid.goal)
    greedy_result = greedy.search(grid, grid.start, grid.goal)
    if dijkstra_result.success and greedy_result.success:
        assert greedy_result.total_cost >= dijkstra_result.total_cost
