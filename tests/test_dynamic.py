"""Tests for the optional dynamic re-planning layer."""

from __future__ import annotations

import pytest

from vector.algorithms import astar
from vector.dynamic import DynamicSimulation
from vector.environment import from_ascii


def make_sim(lines: list[str]) -> DynamicSimulation:
    grid = from_ascii(lines)
    return DynamicSimulation(
        grid=grid,
        search_fn=lambda g, s, t: astar.search(g, s, t),
        agent_pos=grid.start,
        goal=grid.goal,
    )


def test_initial_plan_succeeds_on_open_corridor():
    sim = make_sim(["S....E"])
    result = sim.plan()
    assert result.success
    assert sim.current_path[0] == sim.agent_pos
    assert sim.current_path[-1] == sim.goal


def test_step_moves_agent_along_path():
    sim = make_sim(["S....E"])
    sim.plan()
    sim.step()
    assert sim.agent_pos == (0, 1)


def test_reaches_goal_eventually():
    sim = make_sim(["S....E"])
    sim.plan()
    reached = False
    for _ in range(10):
        reached = sim.step()
        if reached:
            break
    assert reached
    assert sim.agent_pos == sim.goal


def test_obstacle_triggers_replan():
    # Two routes exist; block the planned one and confirm a re-plan happens.
    lines = [
        "S....",
        ".###.",
        "....E",
    ]
    sim = make_sim(lines)
    sim.plan()
    original_path = list(sim.current_path)

    # Block a cell on the current path ahead of the agent.
    block_target = original_path[1]
    sim.toggle_obstacle(block_target)

    assert sim.path_is_valid() is False
    sim.step()  # should trigger an internal re-plan
    assert sim.replan_count == 1


def test_cannot_toggle_agent_or_goal_position():
    sim = make_sim(["S....E"])
    with pytest.raises(ValueError):
        sim.toggle_obstacle(sim.agent_pos)
    with pytest.raises(ValueError):
        sim.toggle_obstacle(sim.goal)


def test_raises_when_fully_sealed_off():
    lines = [
        "S#E",
    ]
    sim = make_sim(lines)
    with pytest.raises(RuntimeError):
        sim.step()
